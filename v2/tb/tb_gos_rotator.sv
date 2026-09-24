// GOS_UNIT_TB: gos_pkg.sv gos_rotator.sv
`timescale 1ns/1ps
// tb_gos_rotator — self-checking TB for gos_rotator.
//  1. unit vectors $VEC_DIR/unit/rotator/{rot_banks,rot_kx,rot_rows}.hex, back-to-back
//  2. exhaustive kx 0..7 x N_RAND random bank words per kx, random valid gaps
//  3. token passthrough and fixed latency L_ROT = 1 (expected-output queue per cycle)
module tb_gos_rotator;
  import gos_pkg::*;

  localparam int TW     = 16;
  localparam int L_ROT  = 1;
  localparam int N_RAND = 2000;
  localparam int MAXV   = 8192;

  logic clk = 1'b0;
  logic rst;
  logic in_valid;
  logic [TW-1:0] in_tok;
  logic [N-1:0][7:0] in_banks;
  logic [2:0] in_kx;
  logic out_valid;
  logic [TW-1:0] out_tok;
  logic [N-1:0][7:0] out_rows;

  gos_rotator #(.TW(TW)) dut (.*);

  always #2.5 clk = ~clk;

  // expected pipeline (model of latency L_ROT)
  logic              exp_v   [L_ROT+1];
  logic [TW-1:0]     exp_tok [L_ROT+1];
  logic [N-1:0][7:0] exp_rows[L_ROT+1];

  int checks = 0;
  int errors = 0;

  function automatic logic [N-1:0][7:0] rot_ref(input logic [N-1:0][7:0] b, input int kx);
    logic [N-1:0][7:0] r;
    for (int i = 0; i < N; i++) r[i] = b[(i + kx) % N];
    return r;
  endfunction

  task automatic fail(input string msg);
    errors++;
    $display("ERROR %s (t=%0t)", msg, $time);
    if (errors > 20) begin
      $display("TEST FAILED too many errors");
      $fatal(1, "TEST FAILED");
    end
  endtask

  // Drive one cycle: set inputs, advance the model, check outputs after the edge.
  task automatic cycle(input logic v, input logic [N-1:0][7:0] banks, input int kx,
                       input logic [N-1:0][7:0] expected);
    in_valid = v;
    in_banks = banks;
    in_kx    = 3'(kx);
    in_tok   = TW'($urandom);
    exp_v[0]    = v;
    exp_tok[0]  = in_tok;
    exp_rows[0] = expected;
    @(posedge clk);
    for (int s = L_ROT; s > 0; s--) begin
      exp_v[s] = exp_v[s-1]; exp_tok[s] = exp_tok[s-1]; exp_rows[s] = exp_rows[s-1];
    end
    #1;
    if (out_valid !== exp_v[L_ROT]) fail($sformatf("out_valid %b exp %b", out_valid, exp_v[L_ROT]));
    else if (exp_v[L_ROT]) begin
      checks++;
      if (out_rows !== exp_rows[L_ROT])
        fail($sformatf("rows %h exp %h", out_rows, exp_rows[L_ROT]));
      if (out_tok !== exp_tok[L_ROT])
        fail($sformatf("tok %h exp %h", out_tok, exp_tok[L_ROT]));
    end
  endtask

  logic [63:0] v_banks [MAXV];
  logic [7:0]  v_kx    [MAXV];
  logic [63:0] v_rows  [MAXV];
  string vec_dir;
  string fname;
  int nvec;

  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) begin
      $display("TEST FAILED no +VEC_DIR");
      $fatal(1, "TEST FAILED");
    end
    for (int i = 0; i < MAXV; i++) begin v_banks[i] = 'x; v_kx[i] = 'x; v_rows[i] = 'x; end
    fname = {vec_dir, "/unit/rotator/rot_banks.hex"}; $readmemh(fname, v_banks);
    fname = {vec_dir, "/unit/rotator/rot_kx.hex"};    $readmemh(fname, v_kx);
    fname = {vec_dir, "/unit/rotator/rot_rows.hex"};  $readmemh(fname, v_rows);
    nvec = 0;
    while (nvec < MAXV && !$isunknown(v_banks[nvec])) nvec++;
    if (nvec == 0) begin
      $display("TEST FAILED no rotator vectors");
      $fatal(1, "TEST FAILED");
    end

    for (int s = 0; s <= L_ROT; s++) exp_v[s] = 1'b0;
    rst = 1'b1; in_valid = 1'b0; in_banks = '0; in_kx = '0; in_tok = '0;
    repeat (3) @(posedge clk);
    #1;
    if (out_valid !== 1'b0) fail("out_valid not 0 in reset");
    rst = 1'b0;

    // 1. unit vectors, back-to-back; also check the TB reference against the file
    for (int i = 0; i < nvec; i++) begin
      if ($isunknown(v_kx[i]) || $isunknown(v_rows[i])) begin
        $display("TEST FAILED malformed vector line %0d", i);
        $fatal(1, "TEST FAILED");
      end
      if (rot_ref(v_banks[i], int'(v_kx[i])) !== v_rows[i])
        fail($sformatf("vector %0d: TB reference disagrees with rot_rows.hex", i));
      cycle(1'b1, v_banks[i], int'(v_kx[i]), v_rows[i]);
    end
    $display("unit vectors: %0d", nvec);

    // 2. exhaustive kx x random data, random gaps; distinct-byte pattern too
    for (int kx = 0; kx < N; kx++) begin
      logic [N-1:0][7:0] b;
      for (int i = 0; i < N; i++) b[i] = 8'(i);
      cycle(1'b1, b, kx, rot_ref(b, kx));
      for (int n = 0; n < N_RAND; n++) begin
        b = {$urandom, $urandom};
        cycle(($urandom % 5) != 0, b, kx, rot_ref(b, kx));
      end
    end
    // random kx interleaved per cycle
    for (int n = 0; n < 4 * N_RAND; n++) begin
      logic [N-1:0][7:0] b;
      int kx;
      b = {$urandom, $urandom};
      kx = $urandom % N;
      cycle(($urandom % 4) != 0, b, kx, rot_ref(b, kx));
    end
    // drain the pipeline
    for (int n = 0; n < L_ROT + 2; n++) cycle(1'b0, '0, 0, '0);

    // reset mid-stream clears valid
    in_valid = 1'b1; rst = 1'b1;
    @(posedge clk); #1;
    if (out_valid !== 1'b0) fail("out_valid not cleared by rst");
    rst = 1'b0; in_valid = 1'b0;

    if (errors == 0) $display("TEST PASSED checks=%0d", checks);
    else begin
      $display("TEST FAILED errors=%0d", errors);
      $fatal(1, "TEST FAILED");
    end
    $finish;
  end
endmodule
