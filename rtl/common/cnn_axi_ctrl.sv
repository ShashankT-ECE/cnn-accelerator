//==============================================================================
// cnn_axi_ctrl.sv — AXI4-Lite control plane for the Phase 2 research accelerator
//
// Wraps rtl/common/cnn_accelerator_v2.sv (the frozen-array reconfigurable
// OS/WS accelerator) behind an AXI4-Lite slave so a host CPU (AMD Kria KV260
// PS ARM Cortex-A53, or any AXI master) can:
//
//   * write the input image (784 B) and Conv1 weights (200 B) through
//     auto-increment write windows,
//   * select the dataflow (0=OS, 1=WS) and commit it at runtime,
//   * start a frame, poll/bring-up on DONE, and read the per-run cycle count
//     and MAC/zero-skip counters,
//   * read the 6,272 int32 results in canonical (ch,y,x) flat order.
//
// The AXI4-Lite slave is single-outstanding, 32-bit, and does not use burst.
// The result path is 8 single-write-port BRAMs (one per pixel-column residue
// x mod 8) written one result word (8 pixels) per cycle by the accelerator —
// no FIFO, no backpressure, no overflow — and read back through one 8192-word
// AXI window.
//
// Register map (byte offsets; all 32-bit, read = RO, write = WO/RW):
//   0x0000 VERSION        RO    [31:16]=ID 0x0002, [15:8]=major 0, [7:0]=minor 1
//   0x0004 CONTROL        WO    [0]=START, [1]=SOFT_RESET, [2]=MODE_COMMIT
//   0x0008 STATUS         RO    [0]=IDLE,[1]=BUSY,[2]=DONE,[3]=ERROR,[4]=SWITCHING
//   0x000C DATAFLOW_MODE  RW    [0] 0=OS, 1=WS
//   0x0010 MODE_STATUS    RO    [0]=ACTIVE_MODE
//   0x0014 SPARSITY_DISABLE RW   [0] 1 = force WS dense (disable coarse skip)
//   0x0018 CYCLE_RUN      RO    per-run cycle count (frozen at DONE)
//   0x001C TOTAL_MACS     RO    156,800
//   0x0020 EXECUTED_MACS  RO    non-zero-activation MACs
//   0x0024 SKIPPED_MACS   RO    zero-activation MACs
//   0x0028 ZERO_SKIP_CYCLES RO  cycles the (tied-0) zero_skip gate is asserted
//   0x002C IMAGE_ADDR     RW    next image write address (0..783)
//   0x0030 IMAGE_DATA     WO    write img[addr], auto-increment addr
//   0x0034 WEIGHT_ADDR    RW    next weight write address (0..199)
//   0x0038 WEIGHT_DATA    WO    write wgt[addr], auto-increment addr
//   0x0040 IRQ_ENABLE     RW    [0]=DONE IRQ enable
//   0x0044 IRQ_STATUS     RO/W1C [0]=DONE pending
//   0x1000..0x8FFF        RO    result window: 8 banks x 1024 x 32-bit;
//                              bank = (byte_addr-0x1000)>>12, dense = (..>>2)&0x3FF
//
// Frozen modules are untouched: pe.sv, systolic_array.sv, input_feed.sv,
// pe_v2.sv, systolic_array_v2.sv. cnn_accelerator_v2.sv is Phase-2 integration
// logic (not frozen) and is used unchanged except for the added
// `sparsity_disable` control.
//==============================================================================

module cnn_axi_ctrl #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 32
) (
    // ---- Clock / reset (active-high synchronous reset, converted by top) ----
    input  logic                      clk,
    input  logic                      rst,

    // ---- AXI4-Lite slave ------------------------------------------------
    input  logic [AXI_ADDR_W-1:0]     s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [AXI_DATA_W-1:0]     s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0]   s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    input  logic [AXI_ADDR_W-1:0]     s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    output logic [AXI_DATA_W-1:0]     s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // ---- Interrupt (level; asserted while DONE pending and enabled) ----
    output logic                      irq
);

    //==========================================================================
    // Register address decode (word indices)
    //==========================================================================
    localparam [5:0] REG_VERSION      = 6'd0;   // 0x00
    localparam [5:0] REG_CONTROL      = 6'd1;   // 0x04
    localparam [5:0] REG_STATUS       = 6'd2;   // 0x08
    localparam [5:0] REG_MODE         = 6'd3;   // 0x0C
    localparam [5:0] REG_MODE_STATUS  = 6'd4;   // 0x10
    localparam [5:0] REG_SPARSITY     = 6'd5;   // 0x14
    localparam [5:0] REG_CYCLE_RUN    = 6'd6;   // 0x18
    localparam [5:0] REG_TOTAL_MACS   = 6'd7;   // 0x1C
    localparam [5:0] REG_EXECUTED     = 6'd8;   // 0x20
    localparam [5:0] REG_SKIPPED      = 6'd9;   // 0x24
    localparam [5:0] REG_ZERO_SKIP    = 6'd10;  // 0x28
    localparam [5:0] REG_IMAGE_ADDR   = 6'd11;  // 0x2C
    localparam [5:0] REG_IMAGE_DATA   = 6'd12;  // 0x30
    localparam [5:0] REG_WEIGHT_ADDR  = 6'd13;  // 0x34
    localparam [5:0] REG_WEIGHT_DATA  = 6'd14;  // 0x38
    localparam [5:0] REG_IRQ_ENABLE   = 6'd16;  // 0x40
    localparam [5:0] REG_IRQ_STATUS   = 6'd17;  // 0x44

    localparam [31:0] VERSION_VALUE   = 32'h0002_0001;

    //==========================================================================
    // Write channel (single outstanding; independent AW/W capture)
    //==========================================================================
    logic                       aw_pending, w_pending;
    logic [AXI_ADDR_W-1:0]      aw_addr_r;
    logic [AXI_DATA_W-1:0]      w_data_r;

    assign s_axi_awready = !aw_pending;
    assign s_axi_wready  = !w_pending;
    assign s_axi_bresp   = 2'b00;

    logic                       write_fire;
    assign write_fire = aw_pending && w_pending;

    //==========================================================================
    // Read channel (2-cycle latency: address capture -> BRAM read -> response)
    //==========================================================================
    typedef enum logic [1:0] {RD_IDLE, RD_LAT, RD_RESP} rd_state_t;
    rd_state_t                  rd_state;
    logic [AXI_ADDR_W-1:0]      rd_addr_r;

    assign s_axi_arready = (rd_state == RD_IDLE);
    assign s_axi_rvalid  = (rd_state == RD_RESP);
    assign s_axi_rresp   = 2'b00;

    //==========================================================================
    // Config / status registers
    //==========================================================================
    logic                       dataflow_mode_r;
    logic                       sparsity_disable_r;
    logic [9:0]                 img_addr_r;
    logic [7:0]                 wgt_addr_r;
    logic                       irq_enable_r;
    logic                       done_sticky;
    logic                       error_sticky;
    logic                       done_irq_pending;

    // Control pulses (1 cycle)
    logic                       start_pulse;
    logic                       soft_reset_pulse;
    logic                       mode_commit_pulse;

    // Image/weight write-port capture (registered, 1-cycle en)
    logic [9:0]                 img_wr_addr_cap;
    logic signed [7:0]         img_wr_data_cap;
    logic                       img_wr_en_r;
    logic [7:0]                 wgt_wr_addr_cap;
    logic signed [7:0]         wgt_wr_data_cap;
    logic                       wgt_wr_en_r;

    // Accelerator control/status
    logic                       acc_busy, acc_done, acc_mode_active;
    logic                       acc_mode_error, acc_switching;
    logic [31:0]                acc_total, acc_exec, acc_skip, acc_zero_skip, acc_cycles;
    logic                       acc_rst;

    assign acc_rst = rst || soft_reset_pulse;

    always_ff @(posedge clk) begin
        if (rst) begin
            aw_pending        <= 1'b0;
            w_pending         <= 1'b0;
            s_axi_bvalid      <= 1'b0;
            dataflow_mode_r   <= 1'b0;
            sparsity_disable_r<= 1'b0;
            img_addr_r        <= 10'd0;
            wgt_addr_r        <= 8'd0;
            irq_enable_r      <= 1'b0;
            done_sticky       <= 1'b0;
            error_sticky      <= 1'b0;
            done_irq_pending  <= 1'b0;
            start_pulse       <= 1'b0;
            soft_reset_pulse  <= 1'b0;
            mode_commit_pulse <= 1'b0;
            img_wr_en_r       <= 1'b0;
            wgt_wr_en_r       <= 1'b0;
            img_wr_addr_cap   <= 10'd0;
            img_wr_data_cap   <= 8'sd0;
            wgt_wr_addr_cap   <= 8'd0;
            wgt_wr_data_cap   <= 8'sd0;
        end else begin
            // default: deassert single-cycle pulses / write-enables
            start_pulse       <= 1'b0;
            soft_reset_pulse  <= 1'b0;
            mode_commit_pulse <= 1'b0;
            img_wr_en_r       <= 1'b0;
            wgt_wr_en_r       <= 1'b0;

            // capture AW / W independently
            if (s_axi_awvalid && s_axi_awready) begin
                aw_addr_r  <= s_axi_awaddr;
                aw_pending <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_data_r   <= s_axi_wdata;
                w_pending  <= 1'b1;
            end

            // apply a completed write
            if (write_fire) begin
                aw_pending <= 1'b0;
                w_pending  <= 1'b0;
                s_axi_bvalid <= 1'b1;
                case (aw_addr_r[5:2])   // word index within the register block
                    REG_CONTROL: begin
                        start_pulse       <= w_data_r[0];
                        soft_reset_pulse  <= w_data_r[1];
                        mode_commit_pulse <= w_data_r[2];
                        if (w_data_r[0]) done_sticky <= 1'b0;   // new run clears DONE
                    end
                    REG_MODE:       dataflow_mode_r <= w_data_r[0];
                    REG_SPARSITY:   sparsity_disable_r <= w_data_r[0];
                    REG_IMAGE_ADDR: img_addr_r <= w_data_r[9:0];
                    REG_IMAGE_DATA: begin
                        img_wr_en_r     <= 1'b1;
                        img_wr_addr_cap <= img_addr_r;
                        img_wr_data_cap <= $signed(w_data_r[7:0]);
                        img_addr_r      <= img_addr_r + 10'd1;
                    end
                    REG_WEIGHT_ADDR: wgt_addr_r <= w_data_r[7:0];
                    REG_WEIGHT_DATA: begin
                        wgt_wr_en_r     <= 1'b1;
                        wgt_wr_addr_cap <= wgt_addr_r;
                        wgt_wr_data_cap <= $signed(w_data_r[7:0]);
                        wgt_addr_r      <= wgt_addr_r + 8'd1;
                    end
                    REG_IRQ_ENABLE: irq_enable_r <= w_data_r[0];
                    REG_IRQ_STATUS: done_irq_pending <= done_irq_pending & ~w_data_r[0]; // W1C
                    default: ;
                endcase
            end

            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            // accelerator status latching
            if (acc_done) begin
                done_sticky      <= 1'b1;
                done_irq_pending <= 1'b1;
            end
            if (acc_mode_error) error_sticky <= 1'b1;
            if (start_pulse) error_sticky <= 1'b0;
        end
    end

    //==========================================================================
    // Read channel FSM
    //==========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            rd_state  <= RD_IDLE;
            rd_addr_r <= '0;
        end else begin
            case (rd_state)
                RD_IDLE: if (s_axi_arvalid && s_axi_arready) begin
                             rd_state  <= RD_LAT;
                             rd_addr_r <= s_axi_araddr;
                         end
                RD_LAT:  rd_state <= RD_RESP;
                RD_RESP: if (s_axi_rready) rd_state <= RD_IDLE;
                default: rd_state <= RD_IDLE;
            endcase
        end
    end

    //==========================================================================
    // Result storage: 8 single-write-port BRAM banks, one per column residue
    // (x mod 8). The accelerator emits one result word (8 pixels + base) per
    // cycle; pixel c maps to canonical flat index `result_base + 7 - c`, hence
    // to bank (7-c) and dense address flat>>3. AXI reads pixel (ch,y,x) from
    // bank (x&7), dense (ch*784+y*28+x)>>3.
    //==========================================================================
    logic               result_valid;
    logic signed [31:0] result_data [0:7];
    logic [12:0]        result_base;

    // The group's leftmost-x is left = result_base % 28 (784 = 28*28 and
    // y*28 ≡ 0 mod 28, so result_base = ch*784 + y*28 + left). Pixel c maps to
    // x = left + 7 - c; it is valid iff 0 <= x <= 27, i.e. left <= 20 + c
    // (the lower bound left+7 >= c holds because left >= 0 and c <= 7).
    logic [4:0]         rleft;
    assign rleft = result_base % 28;

    logic               valid_c [0:7];
    logic [9:0]         dense_c [0:7];
    genvar c;
    generate
        for (c = 0; c < 8; c++) begin : result_decode
            assign valid_c[c] = (rleft <= (20 + c));
            assign dense_c[c] = (result_base + 7 - c) >> 3;
        end
    endgenerate

    // AXI read decode. The result window occupies word addresses 1024..9215
    // (byte 0x1000..0x8FFF); within it, bank = (word-1024)>>10 (0..7) and
    // dense = (word-1024) & 0x3FF.
    logic [13:0]        rd_word;
    logic [2:0]         rd_bank;
    logic [9:0]         rd_dense;
    assign rd_word  = rd_addr_r[15:2];
    assign rd_bank  = (rd_word - 14'd1024) >> 10;
    assign rd_dense = (rd_word - 14'd1024) & 14'd1023;

    logic [31:0]        result_bram_rdata [0:7];
    genvar b;
    generate
        for (b = 0; b < 8; b++) begin : result_bank
            (* ram_style = "block" *) logic [31:0] mem [0:1023];
            // write port: pixel c = 7 - b
            wire               wen = result_valid && valid_c[7-b];
            // read port: AXI (reads every cycle; ignored outside the window)
            always_ff @(posedge clk) begin
                if (wen)
                    mem[dense_c[7-b]] <= result_data[7-b];
                result_bram_rdata[b] <= mem[rd_dense];
            end
        end
    endgenerate

    //==========================================================================
    // Read data mux
    //==========================================================================
    logic [5:0]        rd_reg;
    assign rd_reg = rd_addr_r[5:2];

    always_comb begin
        s_axi_rdata = 32'd0;
        if (rd_word >= 14'd1024 && rd_word < 14'd9216)
            s_axi_rdata = result_bram_rdata[rd_bank];
        else begin
            case (rd_reg)
                REG_VERSION:      s_axi_rdata = VERSION_VALUE;
                REG_STATUS:       s_axi_rdata = {27'b0, acc_switching, error_sticky,
                                                 done_sticky, acc_busy,
                                                 !acc_busy && !acc_switching};
                REG_MODE:         s_axi_rdata = {31'b0, dataflow_mode_r};
                REG_MODE_STATUS:  s_axi_rdata = {31'b0, acc_mode_active};
                REG_SPARSITY:     s_axi_rdata = {31'b0, sparsity_disable_r};
                REG_CYCLE_RUN:    s_axi_rdata = acc_cycles;
                REG_TOTAL_MACS:   s_axi_rdata = acc_total;
                REG_EXECUTED:     s_axi_rdata = acc_exec;
                REG_SKIPPED:      s_axi_rdata = acc_skip;
                REG_ZERO_SKIP:    s_axi_rdata = acc_zero_skip;
                REG_IMAGE_ADDR:   s_axi_rdata = {22'b0, img_addr_r};
                REG_WEIGHT_ADDR:  s_axi_rdata = {24'b0, wgt_addr_r};
                REG_IRQ_ENABLE:   s_axi_rdata = {31'b0, irq_enable_r};
                REG_IRQ_STATUS:   s_axi_rdata = {31'b0, done_irq_pending};
                default:          s_axi_rdata = 32'd0;
            endcase
        end
    end

    //==========================================================================
    // Interrupt
    //==========================================================================
    assign irq = irq_enable_r && done_irq_pending;

    //==========================================================================
    // Phase-2 research accelerator (frozen-array reconfigurable OS/WS)
    //==========================================================================
    cnn_accelerator_v2 u_accel (
        .clk             (clk),
        .rst             (acc_rst),
        .dataflow_mode   (dataflow_mode_r),
        .mode_commit     (mode_commit_pulse),
        .start           (start_pulse),
        .sparsity_disable(sparsity_disable_r),
        .busy            (acc_busy),
        .done            (acc_done),
        .mode_active     (acc_mode_active),
        .mode_error      (acc_mode_error),
        .switching       (acc_switching),
        .img_wr_addr     (img_wr_addr_cap),
        .img_wr_data     (img_wr_data_cap),
        .img_wr_en       (img_wr_en_r),
        .wgt_wr_addr     (wgt_wr_addr_cap),
        .wgt_wr_data     (wgt_wr_data_cap),
        .wgt_wr_en       (wgt_wr_en_r),
        .result_valid    (result_valid),
        .result_data     (result_data),
        .result_base     (result_base),
        .result_last     (),
        .total_macs      (acc_total),
        .executed_macs   (acc_exec),
        .skipped_macs    (acc_skip),
        .zero_skip_cycles(acc_zero_skip),
        .cycle_count     (acc_cycles)
    );

endmodule
