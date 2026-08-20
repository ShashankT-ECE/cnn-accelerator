//==============================================================================
// input_feed_v2.sv — V2 Weight-Stationary Controller / Input-Feed
//
// Drives the frozen V2 systolic array (rtl/common/systolic_array_v2.sv) through
// the audited 42-cycle/group Weight-Stationary schedule
// (docs/specs/INPUT_FEED_V2_SPEC.md, Decision 13). It realizes Conv1
// (1x28x28 -> 6x24x24, valid, no padding) as 6 serialized channels x 72 groups
// (24 rows x 3 groups of 8 output pixels), 4 tiles of 8+8+8+1 taps each.
//
// Frozen schedule (INPUT_FEED_V2_SPEC.md §3/§5):
//   42 cycles/group; accum_clear @0; weight_load @0/15/23/31;
//   result_req[7] @41 (and @0 of the next group, to read the latched result);
//   row-major tap flattening k = 5*ky + kx; eight per-row diagonal-skewed
//   activation streams (collapsed to 5 underlying input-row streams, §7.2).
//
// ACTIVATION BUFFERING (timing fix):
//   The image arrives via a 1-pixel/cycle read port (img_addr/img_data) and is
//   loaded once into 5 block-RAM copies at frame start (784 cycles). During
//   compute the 5 streams read the BRAMs with registered addresses (1-cycle
//   latency), which is compensated by presenting each stream's address one cycle
//   early (a look-ahead of the group/row counters). This removes the ~784:1
//   combinational image mux (previously the 12-level critical path) from the
//   activation path; the address is a small counter+clamp and the BRAM output is
//   registered.
//
// pe_v2.sv and systolic_array_v2.sv are FROZEN and unchanged.
//
// Authoritative contract: docs/specs/INPUT_FEED_V2_SPEC.md (V2 WS).
//==============================================================================

module input_feed_v2 (
    input  logic               clk,
    input  logic               rst,              // synchronous, active-high
    input  logic               start,            // 1-cycle pulse to run full Conv1

    // Image memory (1-pixel/cycle read port; external storage, §12/§16)
    output logic [9:0]         img_addr,         // 0..783 = row*28 + col
    input  logic signed [7:0]  img_data,         // pixel at img_addr
    // Weight memory (combinational read: 6 channels x 5x5, external storage)
    input  logic signed [7:0]  wgt [0:5][0:4][0:4],

    // Array control (drives systolic_array_v2.sv)
    output logic signed [7:0]  act_out      [0:7], // per-row activation streams
    output logic signed [7:0]  w_out        [0:7], // per-row weight (held per tile)
    output logic               weight_load,        // @ cycles 0/15/23/31
    output logic               accum_clear,        // @ cycle 0 only
    output logic               zero_skip,          // tied 0 (WS)
    output logic               dataflow_mode,      // tied 1 (WS)
    output logic               result_req    [0:7],// row 7 @41 (+@0 to read latch)

    // Array data (from systolic_array_v2.sv)
    input  logic signed [31:0] result_in     [0:7],// row-7 result_out (8 cols)

    // Captured results (registered; 8 output pixels per group)
    output logic               done,               // high after last group captured
    output logic               result_valid,       // 1-cycle pulse per group capture
    output logic signed [31:0] result_out    [0:7] // captured result_in snapshot
);

    //--------------------------------------------------------------------------
    // Fixed V2 geometry (INPUT_FEED_V2_SPEC.md §2/§14)
    //--------------------------------------------------------------------------
    localparam int IMG_H   = 28;    // input height
    localparam int IMG_W   = 28;    // input width
    localparam int IMG_N   = 784;   // total pixels (28*28)

    // FSM phases
    localparam logic [1:0] PH_IDLE    = 2'd0;
    localparam logic [1:0] PH_LOAD    = 2'd1;   // load image into BRAM (784 cyc)
    localparam logic [1:0] PH_COMPUTE = 2'd2;   // 432 groups x 42 cycles

    //--------------------------------------------------------------------------
    // Controller state (INPUT_FEED_V2_SPEC.md §4)
    //--------------------------------------------------------------------------
    logic [1:0] phase;
    logic [9:0] load_cnt;   // 0..783 during LOAD

    logic [5:0] s;          // cycle within group (0..41)
    logic [1:0] b;          // group within output row (0..2)
    logic [4:0] y;          // output row (0..23)
    logic [2:0] ch;         // output channel (0..5)
    logic       final_cap;  // 1 during the one final capture cycle
    logic       first;      // 1 until the first group completes (gates result_valid)
    logic       done_r;     // frame complete

    logic       compute;    // == (phase == PH_COMPUTE)
    assign compute = (phase == PH_COMPUTE);

    //--------------------------------------------------------------------------
    // Frame FSM (load phase + group counters; no division)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            phase    <= PH_IDLE;
            load_cnt <= 10'd0;
            s        <= 6'd0;
            b        <= 2'd0;
            y        <= 5'd0;
            ch       <= 3'd0;
            final_cap<= 1'b0;
            done_r   <= 1'b0;
        end else begin
            case (phase)
                PH_IDLE: begin
                    if (start) begin
                        phase    <= PH_LOAD;
                        load_cnt <= 10'd0;
                        s        <= 6'd0;
                        b        <= 2'd0;
                        y        <= 5'd0;
                        ch       <= 3'd0;
                        final_cap<= 1'b0;
                        done_r   <= 1'b0;
                    end
                end

                PH_LOAD: begin
                    if (load_cnt == 10'd783) begin
                        phase <= PH_COMPUTE;
                        s     <= 6'd0;
                        b     <= 2'd0;
                        y     <= 5'd0;
                        ch    <= 3'd0;
                    end else begin
                        load_cnt <= load_cnt + 10'd1;
                    end
                end

                PH_COMPUTE: begin
                    if (final_cap) begin
                        final_cap <= 1'b0;
                        done_r    <= 1'b1;
                        phase     <= PH_IDLE;
                    end else if (s == 6'd41) begin
                        s <= 6'd0;
                        if (b == 2'd2) begin
                            b <= 2'd0;
                            if (y == 5'd23) begin
                                y <= 5'd0;
                                if (ch == 3'd5)
                                    final_cap <= 1'b1;   // last group of frame
                                else
                                    ch <= ch + 3'd1;
                            end else begin
                                y <= y + 5'd1;
                            end
                        end else begin
                            b <= b + 2'd1;
                        end
                    end else begin
                        s <= s + 6'd1;
                    end
                end

                default: phase <= PH_IDLE;
            endcase
        end
    end

    // first: high until the first group completes (suppresses the startup
    // spurious capture at the very first cycle 0).
    always_ff @(posedge clk) begin
        if (rst || (phase == PH_IDLE && start))
            first <= 1'b1;
        else if (compute && s == 6'd41)
            first <= 1'b0;
    end

    //--------------------------------------------------------------------------
    // Image BRAM buffer: 5 copies (one per stream), loaded once at frame start.
    //--------------------------------------------------------------------------
    (* ram_style = "block" *) logic signed [7:0] img_buf [0:4][0:IMG_N-1];

    assign img_addr = (phase == PH_LOAD) ? load_cnt : 10'd0;

    // Write (LOAD): write each incoming pixel to all 5 copies at the same index.
    always_ff @(posedge clk) begin
        if (phase == PH_LOAD) begin
            for (int c = 0; c < 5; c++)
                img_buf[c][load_cnt] <= img_data;
        end
    end

    //--------------------------------------------------------------------------
    // Stream address generation (look-ahead by one cycle to compensate the
    // BRAM 1-cycle read latency).
    //
    //   stream_ky(s) = I[y+ky][ base-8 + s - 5*ky ]   (0 when column OOB, gated)
    //
    // Present the address for cycle s+1 at cycle s:
    //   col0(s+1)  = base(s+1) - 7 + s(s+1)
    //   addr_ky    = (y(s+1)+ky)*28 + clamp(col0(s+1) - 5*ky, 0, 27)
    //--------------------------------------------------------------------------
    logic [5:0] s_next;
    logic [1:0] b_next;
    logic [4:0] y_next;
    logic [5:0] s_eff;
    logic [1:0] b_eff;
    logic [4:0] y_eff;

    assign s_next = (s == 6'd41) ? 6'd0 : s + 6'd1;
    assign b_next = (s == 6'd41) ? ((b == 2'd2) ? 2'd0 : b + 2'd1) : b;
    assign y_next = (s == 6'd41 && b == 2'd2)
                    ? ((y == 5'd23) ? 5'd0 : y + 5'd1) : y;

    // During compute use the next state; during LOAD the next compute state is
    // (0,0,0), so the first compute address is presented one cycle early.
    assign s_eff = compute ? s_next : 6'd0;
    assign b_eff = compute ? b_next : 2'd0;
    assign y_eff = compute ? y_next : 5'd0;

    logic [4:0] base_eff;
    assign base_eff = {b_eff, 3'b111};   // 8*b_eff + 7

    logic signed [7:0] col0_eff;
    assign col0_eff = $signed({3'b0, base_eff}) - 8'sd8 + $signed({2'b0, s_eff});

    logic [9:0] row_base;
    assign row_base = y_eff * IMG_W;   // 5-bit * 28 -> 10-bit

    logic [9:0] bram_addr [0:4];
    always_comb begin
        for (int ky = 0; ky < 5; ky++) begin
            logic signed [7:0] col_ky;
            logic        [4:0] col_clamped;
            col_ky = col0_eff - 5 * ky;
            if (col_ky < 0)       col_clamped = 5'd0;
            else if (col_ky > 27) col_clamped = 5'd27;
            else                  col_clamped = col_ky[4:0];
            bram_addr[ky] = row_base + (ky * IMG_W) + col_clamped;
        end
    end

    // Read (COMPUTE): 5 synchronous reads (registered output).
    logic signed [7:0] stream [0:4];
    always_ff @(posedge clk) begin
        for (int ky = 0; ky < 5; ky++)
            stream[ky] <= img_buf[ky][bram_addr[ky]];
    end

    //--------------------------------------------------------------------------
    // Array control decode (combinational; INPUT_FEED_V2_SPEC.md §5)
    //--------------------------------------------------------------------------
    assign zero_skip     = 1'b0;
    assign dataflow_mode = 1'b1;

    assign accum_clear = compute && !final_cap && (s == 6'd0);
    assign weight_load = compute && !final_cap &&
                         (s == 6'd0 || s == 6'd15 || s == 6'd23 || s == 6'd31);

    genvar rc;
    generate
        for (rc = 0; rc < 8; rc++) begin : req_gen
            assign result_req[rc] = (rc == 7) ?
                (compute && (s == 6'd41 || s == 6'd0)) : 1'b0;
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Weight addressing (INPUT_FEED_V2_SPEC.md §6)
    //--------------------------------------------------------------------------
    logic [1:0] load_tile;
    always_comb begin
        load_tile = 2'd0;
        if      (s == 6'd0)  load_tile = 2'd0;
        else if (s == 6'd15) load_tile = 2'd1;
        else if (s == 6'd23) load_tile = 2'd2;
        else if (s == 6'd31) load_tile = 2'd3;
    end

    genvar wr;
    generate
        for (wr = 0; wr < 8; wr++) begin : wgen
            always_comb begin
                if (load_tile == 2'd3) begin
                    w_out[wr] = (wr == 0) ? wgt[ch][4][4] : 8'sd0;
                end else begin
                    unique case (load_tile)
                        2'd0: w_out[wr] = wgt[ch][(wr)     / 5][(wr)     % 5];
                        2'd1: w_out[wr] = wgt[ch][(8 + wr) / 5][(8 + wr) % 5];
                        2'd2: w_out[wr] = wgt[ch][(16 + wr) / 5][(16 + wr) % 5];
                        default: w_out[wr] = 8'sd0;
                    endcase
                end
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Per-row activation routing (INPUT_FEED_V2_SPEC.md §7.1/§7.2)
    //
    // Row r is in tile t when d = s - r lies in [8t+1, 8t+8]; its tap is
    // k = 8t + r, so ky = k/5. Tile 3 rows 1..7 are pass-through (driven 0).
    //--------------------------------------------------------------------------
    genvar ar;
    generate
        for (ar = 0; ar < 8; ar++) begin : agen
            logic signed [6:0] d;
            logic              active;
            logic        [1:0] tile;
            logic        [2:0] ky_r;

            assign d = $signed({1'b0, s}) - ar;

            always_comb begin
                active = 1'b0;
                tile   = 2'd0;
                if      (d >= 1  && d <= 8)  begin active = 1'b1; tile = 2'd0; end
                else if (d >= 9  && d <= 16) begin active = 1'b1; tile = 2'd1; end
                else if (d >= 17 && d <= 24) begin active = 1'b1; tile = 2'd2; end
                else if (d >= 25 && d <= 32) begin active = 1'b1; tile = 2'd3; end

                if (tile == 2'd3 && ar > 0) active = 1'b0;   // tile-3 pass-through

                unique case (tile)
                    2'd0: ky_r = (ar)      / 5;   // taps 0..7
                    2'd1: ky_r = (8 + ar)  / 5;   // taps 8..15
                    2'd2: ky_r = (16 + ar) / 5;   // taps 16..23
                    2'd3: ky_r = (ar == 0) ? 3'd4 : 3'd0;  // tap 24 (row 0)
                    default: ky_r = 3'd0;
                endcase

                act_out[ar] = active ? stream[ky_r] : 8'sd0;
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Result capture (registered; INPUT_FEED_V2_SPEC.md §10)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int c = 0; c < 8; c++)
                result_out[c] <= 32'sd0;
            result_valid <= 1'b0;
        end else begin
            if (compute && s == 6'd0)
                for (int c = 0; c < 8; c++)
                    result_out[c] <= result_in[c];
            result_valid <= compute && (s == 6'd0) && !first;
        end
    end

    assign done = done_r;

endmodule
