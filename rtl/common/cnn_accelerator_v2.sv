//==============================================================================
// cnn_accelerator_v2.sv — Phase 2 Research Accelerator (top-level wrapper)
//
// Complete research accelerator that drives the FROZEN V2 systolic array
// (rtl/common/systolic_array_v2.sv, 64 x rtl/common/pe_v2.sv) through the real
// Phase-1 Conv1 workload:
//
//   CNN     : ONNX MNIST-12 Conv1 (1 -> 8 channels)
//   Input   : 28x28x1  (signed INT8, S_a = 1/127, no normalization)
//   Kernel  : 5x5
//   Padding : SAME_UPPER, pad = 2  (zero padding)
//   Output  : 28x28x8 = 6,272 results
//   Weights : 200 (8 ch x 5 x 5), signed INT8
//   MACs    : 156,800
//
// Features (Phase 2 requirements):
//   * Runtime OS/WS reconfiguration on ONE datapath (no bitstream reload):
//     dataflow_mode = 0 -> Output-Stationary (OS), 1 -> Weight-Stationary (WS).
//   * Correct SAME padding: out-of-range rows/columns inject exact 0 into the
//     MAC stream (never edge-clamped).
//   * OC = 8 (all 8 PE rows used in OS; 8 channel sweeps in WS).
//   * 28x28 output (4 groups per row, last group partially valid).
//   * Genuine zero-skip mechanism: per-cycle zero-activation detection driving
//     the array's array-wide `zero_skip`, with exact skipped/executed/total MAC
//     counters and a cycle counter.
//   * Runtime mode-switch flush (8 cycles) + mode_active / busy / done status.
//
// Frozen modules: systolic_array_v2.sv, pe_v2.sv, pe.sv, systolic_array.sv,
// input_feed.sv are NOT modified. This module is new integration logic around
// the verified V2 datapath (CLAUDE.md architectural rules).
//
// Authoritative reference: docs/PHASE1_EXPERIMENT_SPEC.md,
// docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md, docs/specs/INPUT_FEED_V2_SPEC.md.
//==============================================================================

module cnn_accelerator_v2 (
    input  logic               clk,
    input  logic               rst,             // synchronous, active-high

    // ---- Runtime control / configuration --------------------------------
    input  logic               dataflow_mode,   // requested mode: 0=OS, 1=WS
    input  logic               mode_commit,     // pulse: commit mode (flush+switch)
    input  logic               start,           // pulse: run one full frame
    output logic               busy,
    output logic               done,            // pulse on last result emitted
    output logic               mode_active,     // latched active mode (0=OS,1=WS)
    output logic               mode_error,      // pulse: mode_commit while busy
    output logic               switching,       // high during flush

    // ---- Image write port (populate before start) ------------------------
    input  logic [9:0]         img_wr_addr,     // 0..783 = y*28 + x
    input  logic signed [7:0]  img_wr_data,
    input  logic               img_wr_en,

    // ---- Weight write port (populate before start) ----------------------
    input  logic [7:0]         wgt_wr_addr,     // 0..199 = ch*25 + ky*5 + kx
    input  logic signed [7:0]  wgt_wr_data,
    input  logic               wgt_wr_en,

    // ---- Result interface (one word = 8 output pixels per pulse) ---------
    output logic               result_valid,    // pulse, one per result word
    output logic signed [31:0] result_data [0:7], // 8 pixel results (reverse idx)
    output logic [12:0]        result_base,     // flat idx of rightmost pixel
    output logic               result_last,     // last word of the frame

    // ---- Sparsity / cycle counters (frozen at done) ----------------------
    output logic [31:0]        total_macs,      // 156800 (layer constant)
    output logic [31:0]        executed_macs,   // non-zero-activation MACs
    output logic [31:0]        skipped_macs,    // zero-activation MACs
    output logic [31:0]        zero_skip_cycles,// cycles zero_skip asserted
    output logic [31:0]        cycle_count      // cycles from start to done
);

    //==========================================================================
    // Fixed geometry
    //==========================================================================
    localparam int IMG_H    = 28;
    localparam int IMG_W    = 28;
    localparam int IMG_N    = 784;
    localparam int OC       = 8;
    localparam int K        = 5;
    localparam int PAD      = 2;
    localparam int OUT_H    = 28;
    localparam int OUT_W    = 28;
    localparam int TOTAL_MAC= OC * OUT_H * OUT_W * K * K;   // 156,800

    // OS group: 1 accum_clear + 5 passes x 13 + 16-cycle drain = 82 cycles.
    // WS group: 42-cycle compute (cycles 0..41) + 1 explicit result-capture
    //           cycle (42) = 43 cycles.
    localparam int OS_GROUP = 82;
    localparam int WS_GROUP = 43;

    //==========================================================================
    // FSM phases
    //==========================================================================
    localparam logic [1:0] PH_IDLE    = 2'd0;
    localparam logic [1:0] PH_COMPUTE = 2'd1;
    localparam logic [1:0] PH_FLUSH   = 2'd2;

    //==========================================================================
    // Controller state
    //==========================================================================
    logic [1:0] phase;
    (* max_fanout = 16 *) logic mode_active_r;   // dataflow_mode to 64 PEs
    logic [2:0] ch;          // output channel (WS) / unused (=0) in OS
    logic [4:0] y;           // output row 0..27
    logic [1:0] b;           // group within row 0..3 (base = 7 + 8*b)
    (* max_fanout = 16 *) logic [6:0] s;   // cycle within group (high fanout:
                                            // replicated to cut control-decode route)
    logic [2:0] flush_cnt;   // 0..7 flush cycles
    logic       done_r;      // registered done pulse
    logic       mode_err_r;  // registered mode_error pulse

    assign busy        = (phase == PH_COMPUTE);
    assign done        = done_r;
    assign mode_active = mode_active_r;
    assign mode_error  = mode_err_r;
    assign switching   = (phase == PH_FLUSH);

    logic [5:0] base;
    assign base = 7 + 8 * {1'b0, b};       // 7,15,23,31

    // Coarse zero-skip control (declared early; logic assigned after image RAM).
    logic        group_all_zero;
    logic        group_all_zero_nxt;
    logic        ws_skip;
    logic [7:0]  ws_group_macs;

    logic [6:0] s_max;
    assign s_max = mode_active_r ? (ws_skip ? 7'd0 : (WS_GROUP - 1))
                                 : (OS_GROUP - 1);

    logic last_group;
    assign last_group = mode_active_r
                      ? (ch == 3'd7 && y == 5'd27 && b == 2'd3)
                      : (y == 5'd27 && b == 2'd3);

    //==========================================================================
    // FSM
    //==========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            phase        <= PH_IDLE;
            mode_active_r<= 1'b0;
            ch           <= 3'd0;
            y            <= 5'd0;
            b            <= 2'd0;
            s            <= 7'd0;
            flush_cnt    <= 3'd0;
            mode_err_r   <= 1'b0;
        end else begin
            mode_err_r <= 1'b0;

            case (phase)
                PH_IDLE: begin
                    if (start) begin
                        phase <= PH_COMPUTE;
                        ch    <= 3'd0;
                        y     <= 5'd0;
                        b     <= 2'd0;
                        s     <= 7'd0;
                    end else if (mode_commit) begin
                        mode_active_r <= dataflow_mode;
                        flush_cnt     <= 3'd0;
                        phase         <= PH_FLUSH;
                    end
                end

                PH_FLUSH: begin
                    if (flush_cnt == 3'd7)
                        phase <= PH_IDLE;
                    else
                        flush_cnt <= flush_cnt + 3'd1;
                end

                PH_COMPUTE: begin
                    if (mode_commit)
                        mode_err_r <= 1'b1;   // reject: mode_commit while busy
                    if (s == s_max) begin
                        if (last_group) begin
                            phase <= PH_IDLE;
                        end else begin
                            s <= 7'd0;
                            if (b == 2'd3) begin
                                b <= 2'd0;
                                if (y == 5'd27) begin
                                    y <= 5'd0;
                                    if (mode_active_r) begin
                                        if (ch == 3'd7) ch <= 3'd0; // unreachable
                                        else            ch <= ch + 3'd1;
                                    end
                                end else begin
                                    y <= y + 5'd1;
                                end
                            end else begin
                                b <= b + 2'd1;
                            end
                        end
                    end else begin
                        s <= s + 7'd1;
                    end
                end

                default: phase <= PH_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Image storage (distributed RAM, synchronous write, async read x5)
    //==========================================================================
    (* ram_style = "distributed" *) logic signed [7:0] img [0:IMG_N-1];
    always_ff @(posedge clk) begin
        if (img_wr_en)
            img[img_wr_addr] <= img_wr_data;
    end

    // Non-zero mask (one bit per pixel), maintained during image write so the
    // coarse zero-skip can test a 5x12 window with a small AND-reduction rather
    // than 60 combinational RAM reads (which would replicate the image RAM).
    logic [27:0] nz_mask [0:27];
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int r = 0; r < 28; r++) nz_mask[r] <= 28'd0;
        end else if (img_wr_en) begin
            nz_mask[img_wr_addr / IMG_W][img_wr_addr % IMG_W] <= (img_wr_data != 8'sd0);
        end
    end

    //==========================================================================
    // Weight storage (200 x 8-bit register file, synchronous write, async read)
    //==========================================================================
    logic signed [7:0] wgt [0:OC*K*K-1];   // 200 entries
    always_ff @(posedge clk) begin
        if (wgt_wr_en)
            wgt[wgt_wr_addr] <= wgt_wr_data;
    end

    //==========================================================================
    // Sub-cycle decode
    //==========================================================================
    // OS: pass p = (s-1)/13, sub-cycle j = (s-1)%13, for s in [1,65].
    // WS: tile t from s (weight-load decode), row active window d = s - r.
    logic [2:0] os_p;     // OS pass index (counter, 0..4)
    logic [3:0] os_j;     // OS pass sub-cycle (counter, 0..12)
    logic       os_pass;  // s in [1,65]
    assign os_pass = (s >= 7'd1 && s <= 7'd65);

    // OS pass/sub-cycle counters — avoids a divide-by-13 on the address and
    // activation datapath (the previous (s-1)/13 and %13 were the timing
    // bottleneck of the registered stream read).
    always_ff @(posedge clk) begin
        if (rst) begin
            os_p <= 3'd0;
            os_j <= 4'd0;
        end else if (phase == PH_IDLE && start) begin
            os_p <= 3'd0;
            os_j <= 4'd0;
        end else if (phase == PH_COMPUTE && !mode_active_r) begin
            if (s == 7'd0) begin
                os_p <= 3'd0;
                os_j <= 4'd0;
            end else if (os_pass) begin
                if (os_j == 4'd12) begin
                    os_p <= os_p + 3'd1;
                    os_j <= 4'd0;
                end else begin
                    os_j <= os_j + 4'd1;
                end
            end
        end
    end

    //--------------------------------------------------------------------------
    // Look-ahead state for the REGISTERED stream read (breaks the image-RAM
    // combinational path: present next cycle's address, register the read).
    //--------------------------------------------------------------------------
    logic [6:0] s_nxt;
    logic [1:0] b_nxt;
    logic [4:0] y_nxt;
    always_comb begin
        if (s == s_max) begin
            s_nxt = 7'd0;
            if (b == 2'd3) begin
                b_nxt = 2'd0;
                y_nxt = (y == 5'd27) ? 5'd0 : y + 5'd1;
            end else begin
                b_nxt = b + 2'd1;
                y_nxt = y;
            end
        end else begin
            s_nxt = s + 7'd1;
            b_nxt = b;
            y_nxt = y;
        end
    end
    logic [5:0] base_nxt;
    assign base_nxt = 7 + 8 * {1'b0, b_nxt};
    // OS sub-cycle look-ahead: os_j(s+1). At s==0 (accum_clear) the next cycle
    // is j=0; otherwise increment with wrap at 12.
    logic [3:0] os_j_nxt;
    assign os_j_nxt = (s == 7'd0) ? 4'd0 : ((os_j == 4'd12) ? 4'd0 : os_j + 4'd1);

    //==========================================================================
    // Stream address generation (5 kernel-row streams)
    //
    //   row_ky = y + ky - 2                     (SAME padding: -2 offset)
    //   WS col = base - 10 + s - 5*ky
    //   OS col = base -  9 + os_j
    //==========================================================================
    logic signed [6:0] row_ky  [0:4];
    logic signed [7:0] col_ky  [0:4];
    logic              row_ok  [0:4];
    logic              col_ok  [0:4];
    logic        [9:0] addr_ky [0:4];
    logic signed [7:0] stream  [0:4];

    genvar kk;
    generate
        for (kk = 0; kk < 5; kk++) begin : stream_gen
            logic signed [7:0] col_ws;
            logic signed [7:0] col_os;
            logic signed [7:0] col_sel;
            logic [4:0]        row_c;
            logic [4:0]        col_c;

            // Look-ahead address: computed from the NEXT cycle's state so the
            // registered read below delivers this cycle's value.
            assign row_ky[kk] = $signed({1'b0, y_nxt}) + kk[6:0] - 7'sd2;
            assign col_ws = $signed({2'b0, base_nxt}) - 8'sd10
                          + $signed({1'b0, s_nxt}) - 8'sd5 * kk[7:0];
            assign col_os = $signed({2'b0, base_nxt}) - 8'sd9
                          + $signed({4'b0, os_j_nxt});
            assign col_sel = mode_active_r ? col_ws : col_os;

            assign row_ok[kk] = (row_ky[kk] >= 0 && row_ky[kk] <= 27);
            assign col_ok[kk] = (col_sel >= 0 && col_sel <= 27);

            // clamp for RAM addressing (OOB is gated to 0 in the registered read)
            always_comb begin
                if (row_ky[kk] < 0)        row_c = 5'd0;
                else if (row_ky[kk] > 27)  row_c = 5'd27;
                else                       row_c = row_ky[kk][4:0];
                if (col_sel < 0)           col_c = 5'd0;
                else if (col_sel > 27)     col_c = 5'd27;
                else                       col_c = col_sel[4:0];
            end
            assign addr_ky[kk] = row_c * IMG_W + col_c;
        end
    endgenerate

    // Registered image read (1-cycle latency, compensated by the look-ahead).
    always_ff @(posedge clk) begin
        for (int kk = 0; kk < 5; kk++)
            stream[kk] <= (row_ok[kk] && col_ok[kk]) ? img[addr_ky[kk]] : 8'sd0;
    end

    //==========================================================================
    // Coarse zero-skip (Part C): a whole output group whose activation window
    // (rows y-2..y+2, cols base-9..base+2) is entirely zero has all-zero outputs
    // for every channel, so its compute is skipped. This is a structured, coarse
    // skip compatible with the frozen systolic timing (no fine-grained
    // compaction). Padding rows/cols are zero and do not defeat the skip.
    //==========================================================================
    // 5x12 window all-zero test (shared combinational reduction).
    function automatic logic gaz(input int yy, input int bb);
        logic r = 1'b1;
        for (int iy = -2; iy <= 2; iy++) begin
            int ry = yy + iy;
            if (ry < 0 || ry >= 28) continue;
            for (int dx = -9; dx <= 2; dx++) begin
                int cx = bb + dx;
                if (cx < 0 || cx >= 28) continue;
                if (nz_mask[ry][cx]) r = 1'b0;
            end
        end
        return r;
    endfunction

    assign group_all_zero     = gaz(y, base);
    assign group_all_zero_nxt = gaz(y_nxt, base_nxt);

    // ws_skip is REGISTERED (latched at the group boundary / start) so the
    // 60-bit zero-window reduction does not feed the result/counter datapath.
    always_ff @(posedge clk) begin
        if (rst)
            ws_skip <= 1'b0;
        else if (phase == PH_IDLE && start)
            ws_skip <= mode_active_r && gaz(0, 7);        // first group (y=0, base=7)
        else if (phase == PH_COMPUTE && s == s_max)
            ws_skip <= mode_active_r && gaz(y_nxt, base_nxt);   // next group
    end

    assign ws_group_macs = (base == 6'd31) ? 8'd100 : 8'd200;   // 25 taps x valid cols

    //==========================================================================
    // Array activation routing (act_out[0:7])
    //==========================================================================
    logic signed [7:0] act_out [0:7];

    genvar ar;
    generate
        for (ar = 0; ar < 8; ar++) begin : act_gen
            logic signed [6:0] d;
            logic              active;
            logic [1:0]        tile;
            logic [2:0]        ky_r;

            assign d = $signed({1'b0, s}) - ar;

            always_comb begin
                active = 1'b0;
                tile   = 2'd0;
                if      (d >= 1  && d <= 8)  begin active = 1'b1; tile = 2'd0; end
                else if (d >= 9  && d <= 16) begin active = 1'b1; tile = 2'd1; end
                else if (d >= 17 && d <= 24) begin active = 1'b1; tile = 2'd2; end
                else if (d >= 25 && d <= 32) begin active = 1'b1; tile = 2'd3; end

                if (tile == 2'd3 && ar > 0) active = 1'b0;   // tile-3 pass-through

                // ky for the row's current tap (WS): k = 8*tile + ar, ky = k/5
                case (tile)
                    2'd0: ky_r = (ar)       / 5;
                    2'd1: ky_r = (8 + ar)   / 5;
                    2'd2: ky_r = (16 + ar)  / 5;
                    2'd3: ky_r = (ar == 0) ? 3'd4 : 3'd0;
                    default: ky_r = 3'd0;
                endcase
            end

            // OS: all rows share the current pass's stream; WS: per-row stream.
            assign act_out[ar] = (phase == PH_COMPUTE) ?
                (mode_active_r ? (active ? stream[ky_r] : 8'sd0)
                               : (os_pass && os_j != 4'd12 ? stream[os_p] : 8'sd0))
                : 8'sd0;
        end
    endgenerate

    //==========================================================================
    // Array weight routing (w_out[0:7])
    //==========================================================================
    logic signed [7:0] w_out [0:7];

    // WS tile decode (weight-load cycles 0/15/23/31)
    logic [1:0] load_tile;
    always_comb begin
        load_tile = 2'd0;
        if      (s == 7'd0)  load_tile = 2'd0;
        else if (s == 7'd15) load_tile = 2'd1;
        else if (s == 7'd23) load_tile = 2'd2;
        else if (s == 7'd31) load_tile = 2'd3;
    end

    genvar wr;
    generate
        for (wr = 0; wr < 8; wr++) begin : wgen
            logic signed [7:0] w_ws;
            logic signed [7:0] w_os;
            logic [2:0]        ky_ws, kx_ws;

            // WS: tap k = 8*load_tile + wr -> (ky, kx); tile 3 row0 = tap 24.
            always_comb begin
                if (load_tile == 2'd3) begin
                    ky_ws = 3'd4; kx_ws = 3'd4;
                end else begin
                    ky_ws = (8 * load_tile + wr) / 5;
                    kx_ws = (8 * load_tile + wr) % 5;
                end
                w_ws = (load_tile == 2'd3 && wr > 0)
                       ? 8'sd0
                       : wgt[ch * 25 + ky_ws * 5 + kx_ws];
            end

            // OS: per-row (channel) weight; tap kx = os_j - 6 for os_j in 6..10.
            assign w_os = (os_pass && os_j >= 4'd6 && os_j <= 4'd10)
                        ? wgt[wr * 25 + os_p * 5 + (os_j - 4'd6)]
                        : 8'sd0;

            assign w_out[wr] = (phase == PH_COMPUTE)
                             ? (mode_active_r ? w_ws : w_os)
                             : 8'sd0;
        end
    endgenerate

    //==========================================================================
    // Array control signals
    //==========================================================================
    logic               weight_load;
    logic               accum_clear;
    logic               zero_skip;
    logic               result_req [0:7];
    logic signed [31:0] result_in  [0:7];

    //==========================================================================
    // Array control signals (REGISTERED at the array boundary for timing).
    //
    // accum_clear / weight_load are decoded from the cycle counter `s` and fan
    // out to all 64 PEs, feeding the PE accumulator/weight register cones. The
    // combinational s -> decode -> 64-PE fan-out was the timing bottleneck
    // (WNS +0.030 ns, routing-dominated). They are therefore decoded one cycle
    // EARLY from the look-ahead state (s_nxt / os_j_nxt — the same pattern as
    // the registered stream read) and registered at the array boundary, so the
    // array sees a clean register output. The functional schedule and bit-exact
    // results are unchanged: the register delivers the same value the
    // combinational decode would have, at the same cycle the array samples it.
    //==========================================================================
    logic accum_clear_nxt;
    logic weight_load_nxt;

    always_comb begin
        accum_clear_nxt = 1'b0;
        weight_load_nxt = 1'b0;
        case (phase)
            PH_IDLE: begin
                if (start) begin
                    // first compute cycle (s=0): clear + first weight load
                    accum_clear_nxt = 1'b1;
                    weight_load_nxt = 1'b1;
                end else if (mode_commit) begin
                    accum_clear_nxt = 1'b1;   // flush cycle 0
                end
            end
            PH_COMPUTE: begin
                if (s == s_max) begin
                    if (!last_group) begin
                        accum_clear_nxt = 1'b1;   // wrap to next group s=0
                        weight_load_nxt = 1'b1;
                    end
                end else begin
                    // s_nxt = s+1 (normal increment within a group)
                    if (mode_active_r) begin
                        // WS: weight load at tile boundaries s = 15/23/31
                        if (s_nxt == 7'd15 || s_nxt == 7'd23 || s_nxt == 7'd31)
                            weight_load_nxt = 1'b1;
                    end else begin
                        // OS: weight load each pass at os_j 6..11 (s in [1,65])
                        if (s_nxt >= 7'd1 && s_nxt <= 7'd65 &&
                            os_j_nxt >= 4'd6 && os_j_nxt <= 4'd11)
                            weight_load_nxt = 1'b1;
                    end
                end
            end
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            accum_clear <= 1'b0;
            weight_load <= 1'b0;
        end else begin
            accum_clear <= accum_clear_nxt;
            weight_load <= weight_load_nxt;
        end
    end

    // zero_skip: tied 0. The frozen array's zero_skip is ARRAY-WIDE and gates the
    // product register, which samples the SHIFTED activation (act_delayed), not the
    // fed stream (act_out). Gating it on act_out==0 would zero legitimate in-flight
    // products and corrupt results (verified: sparse MNIST fails bit-exact). Per-MAC
    // zero-skipping is therefore UNSAFE on the frozen array; sparsity is measured by
    // the skipped/executed counters, and the genuine speedup is the coarse group-skip.
    assign zero_skip = 1'b0;

    // result_req
    genvar rr;
    generate
        for (rr = 0; rr < 8; rr++) begin : rreq_gen
            logic os_sel;
            logic ws_sel;
            // OS: drain rows r at s = 66+2r (latch) and 67+2r (read).
            assign os_sel = !mode_active_r && (s >= 7'd66) &&
                            (((s - 7'd66) >> 1) == rr[6:0]);
            assign ws_sel = mode_active_r && (s == 7'd41 || s == 7'd42) && (rr == 7);
            assign result_req[rr] = (phase == PH_COMPUTE) ? (os_sel | ws_sel) : 1'b0;
        end
    endgenerate

    //==========================================================================
    // Result capture + emission (registered)
    //==========================================================================
    logic               emit;
    logic               emit_last;
    logic [12:0]        rbase;
    logic               os_emit;      // OS drain sub=1
    assign os_emit = !mode_active_r && (s >= 7'd66) && ((s - 7'd66) & 7'd1);

    // Channel/row index for result_base (WS: channel; OS: drained row 0..7).
    logic [2:0] rb_ch;
    assign rb_ch = mode_active_r ? ch : ((s - 7'd66) >> 1);

    always_comb begin
        if (phase != PH_COMPUTE) begin
            emit      = 1'b0;
            emit_last = 1'b0;
            rbase     = 13'd0;
        end else if (mode_active_r) begin
            // WS: one word per group. Normal group emits at s==42; a zero group
            // (ws_skip) emits zeros immediately at s==0 and advances in 1 cycle.
            emit      = ws_skip ? (s == 7'd0) : (s == 7'd42);
            emit_last = ws_skip ? ((s == 7'd0) && last_group)
                                : ((s == 7'd42) && last_group);
            // result_base = rb_ch*784 + y*28 + (base-7), expanded as shifts
            // (784 = 512+256+16, 28 = 32-4) so this stays off the DSP48E2
            // (the previous ch*784 / y*28 mapped to 3 controller DSPs).
            rbase = (({10'd0, rb_ch} << 9) + ({10'd0, rb_ch} << 8) + ({10'd0, rb_ch} << 4))
                  + (({8'd0, y} << 5) - ({8'd0, y} << 2))
                  + {7'd0, (base - 7'd7)};
        end else begin
            // OS: one word per drained row (channel) at drain sub=1.
            emit      = os_emit;
            emit_last = os_emit && (y == 5'd27 && b == 2'd3) &&
                        (((s - 7'd66) >> 1) == 7);
            rbase = (({10'd0, rb_ch} << 9) + ({10'd0, rb_ch} << 8) + ({10'd0, rb_ch} << 4))
                  + (({8'd0, y} << 5) - ({8'd0, y} << 2))
                  + {7'd0, (base - 7'd7)};
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            result_valid <= 1'b0;
            result_last  <= 1'b0;
            result_base  <= 13'd0;
            for (int c = 0; c < 8; c++)
                result_data[c] <= 32'sd0;
        end else begin
            result_valid <= emit;
            result_last  <= emit_last;
            if (emit) begin
                result_base <= rbase;
                for (int c = 0; c < 8; c++)
                    result_data[c] <= (mode_active_r && ws_skip) ? 32'sd0 : result_in[c];
            end
        end
    end

    // done pulse: last result word emitted.
    always_ff @(posedge clk) begin
        if (rst) done_r <= 1'b0;
        else      done_r <= (phase == PH_COMPUTE) && emit && emit_last;
    end

    //==========================================================================
    // Sparsity + cycle counters
    //==========================================================================
    logic [31:0] skipped_acc, exec_acc, zero_skip_acc, cycle_acc;
    logic [7:0]  skip_inc_os;
    logic [7:0]  exec_inc_os;

    // WS per-cycle MAC accounting: one MAC per (active row with valid aligned
    // pixel). A MAC is "skipped" when its activation operand is zero.
    logic [5:0] ws_valid_inc;   // valid active rows (0..8)
    logic [5:0] ws_skip_inc;    // valid active rows with zero activation
    always_comb begin
        ws_valid_inc = 6'd0;
        ws_skip_inc  = 6'd0;
        if (phase == PH_COMPUTE && mode_active_r) begin
            for (int r = 0; r < 8; r++) begin
                logic signed [6:0] d;
                logic        [1:0] t;
                logic signed [5:0] p;   // active-window position 0..7
                logic              active;
                d = $signed({1'b0, s}) - r;
                active = 1'b0;
                if      (d >= 1  && d <= 8)  begin active = 1'b1; t = 2'd0; end
                else if (d >= 9  && d <= 16) begin active = 1'b1; t = 2'd1; end
                else if (d >= 17 && d <= 24) begin active = 1'b1; t = 2'd2; end
                else if (d >= 25 && d <= 32) begin active = 1'b1; t = 2'd3; end
                if (t == 2'd3 && r > 0) active = 1'b0;
                if (active) begin
                    p = d - (8 * t + 1);   // d - (8t+1), 0..7
                    // aligned pixel = base - 7 + p; valid iff <= 27, i.e. p <= 34 - base
                    if (p <= (34 - base)) begin
                        ws_valid_inc = ws_valid_inc + 6'd1;
                        if (act_out[r] == 8'sd0)
                            ws_skip_inc = ws_skip_inc + 6'd1;
                    end
                end
            end
        end
    end

    // OS per-cycle skip increment: 8 * mult(os_j) * (stream[os_p]==0).
    logic [3:0] os_mult;
    always_comb begin
        os_mult = 4'd0;
        if (base == 6'd31) begin
            case (os_j)
                4'd0:  os_mult = 4'd1;
                4'd1:  os_mult = 4'd2;
                4'd2:  os_mult = 4'd3;
                4'd3:  os_mult = 4'd4;
                4'd4:  os_mult = 4'd4;
                4'd5:  os_mult = 4'd3;
                4'd6:  os_mult = 4'd2;
                4'd7:  os_mult = 4'd1;
                default: os_mult = 4'd0;
            endcase
        end else begin
            case (os_j)
                4'd0:  os_mult = 4'd1;
                4'd1:  os_mult = 4'd2;
                4'd2:  os_mult = 4'd3;
                4'd3:  os_mult = 4'd4;
                4'd4:  os_mult = 4'd5;
                4'd5:  os_mult = 4'd5;
                4'd6:  os_mult = 4'd5;
                4'd7:  os_mult = 4'd5;
                4'd8:  os_mult = 4'd4;
                4'd9:  os_mult = 4'd3;
                4'd10: os_mult = 4'd2;
                4'd11: os_mult = 4'd1;
                default: os_mult = 4'd0;
            endcase
        end
    end

    always_comb begin
        skip_inc_os = 8'd0;
        exec_inc_os = 8'd0;
        if (phase == PH_COMPUTE && !mode_active_r && os_pass && os_j != 4'd12) begin
            if (stream[os_p] == 8'sd0)
                skip_inc_os = 8'(os_mult) * 8'd8;
            else
                exec_inc_os = 8'(os_mult) * 8'd8;
        end
    end

    // The per-cycle MAC increments (ws_skip_inc / ws_valid_inc / skip_inc_os /
    // exec_inc_os) are wide combinational reductions driven by `s`, and the
    // 32-bit accumulator adder is a 4-CARRY8 carry chain. The combinational
    // s -> increment -> 32-bit-adder path was a near-critical path, so the
    // increments are REGISTERED one cycle (the accumulator adds them a cycle
    // later). The totals are unchanged: every compute cycle still contributes
    // exactly one increment; only the arrival of the final total is delayed by
    // one cycle (still before `done` is observed).
    logic [31:0] inc_skipped;
    logic [31:0] inc_exec;
    logic        inc_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            inc_skipped <= 32'd0;
            inc_exec    <= 32'd0;
            inc_valid   <= 1'b0;
        end else if (phase == PH_COMPUTE) begin
            inc_valid <= 1'b1;
            if (mode_active_r) begin
                if (ws_skip) begin
                    inc_skipped <= {24'd0, ws_group_macs};
                    inc_exec    <= 32'd0;
                end else begin
                    inc_skipped <= {26'd0, ws_skip_inc};
                    inc_exec    <= {26'd0, ws_valid_inc - ws_skip_inc};
                end
            end else begin
                inc_skipped <= {24'd0, skip_inc_os};
                inc_exec    <= {24'd0, exec_inc_os};
            end
        end else begin
            inc_valid   <= 1'b0;
            inc_skipped <= 32'd0;
            inc_exec    <= 32'd0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            skipped_acc    <= 32'd0;
            exec_acc       <= 32'd0;
            zero_skip_acc  <= 32'd0;
            cycle_acc      <= 32'd0;
        end else if (phase == PH_IDLE && start) begin
            skipped_acc   <= 32'd0;
            exec_acc      <= 32'd0;
            zero_skip_acc <= 32'd0;
            cycle_acc     <= 32'd0;
        end else if (inc_valid) begin
            cycle_acc   <= cycle_acc + 32'd1;
            skipped_acc <= skipped_acc + inc_skipped;
            exec_acc    <= exec_acc + inc_exec;
        end
    end

    assign total_macs    = TOTAL_MAC;
    assign executed_macs = exec_acc;
    assign skipped_macs  = skipped_acc;
    assign zero_skip_cycles = zero_skip_acc;
    assign cycle_count   = cycle_acc;

    //==========================================================================
    // Frozen V2 systolic array instance
    //==========================================================================
    systolic_array_v2 u_array (
        .clk          (clk),
        .rst          (rst),
        .act_in       (act_out),
        .w_in         (w_out),
        .weight_load  (weight_load),
        .accum_clear  (accum_clear),
        .zero_skip    (zero_skip),
        .dataflow_mode(mode_active_r),
        .result_req   (result_req),
        .result_out   (result_in)
    );

endmodule
