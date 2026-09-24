// GOS_UNIT_TB: gos_pkg.sv gos_requant.sv
`timescale 1ns/1ps
// tb_gos_requant — self-checking, bit-exact TB for gos_requant (L_RQ = 6).
//
// Part A (vectors): every row of $VEC_DIR/unit/requant/{lenet5,cifar10}_requant.hex
// (v2/vectors/README.md §3.4, 160-bit rows). Each row is driven in TWO cycles with its
// own (q_bias, m, s): relu_en = 0 (checked against the file's pre-ReLU q) and relu_en = 1
// (checked against the file's q_relu). The row sits in lane (row_index mod 8), so every
// lane carries vector rows; the other 7 lanes carry TB-generated v for the same channel
// params (row v + small delta, exact-region uniform, or |v| < 2^25 uniform), checked
// against the TB reference model. v_raw is checked in every lane (file v for the row lane).
// The TB model is also cross-checked against the file q/q_relu on every row.
// Near-tie rows (flag bit 134) must include (QPARAM idx, v) = lenet5 (104, ±55930)
// [conv5 ch82], (196, ±31044) [fc1 ch54]; cifar10 (121, ±36475) [r2 conv3 ch57].
//
// Part B (random): shared per-cycle m in [2^31, 2^32), s in 1..63, random relu_en,
// lane v with |v| < 2^25 (uniform / exact region / small), forced exact ties
// p = (2k+1)·2^(s-1) (m = mo·2^b, v = u·2^(s-1-b), mo·u odd), and s = 0 cycles
// (m = 0 final-layer style and random m) which must give q = 0.
//
// Stimulus is back-to-back with random idle gaps (in_valid = 0, garbage inputs);
// out_tok/out_valid/out_raw_d and the exact latency L_RQ are checked per item.
// Reference model: exact 128-bit signed arithmetic of the ARCH_SPEC formula.
module tb_gos_requant;
  import gos_pkg::*;

  localparam int TW    = 16;
  localparam int L_RQ  = 6;
  localparam int N_RND = 150000;     // random cycles (x 8 lanes)

  logic                    clk = 1'b0;
  logic                    rst = 1'b1;
  logic                    in_valid = 1'b0;
  logic [TW-1:0]           in_tok = '0;
  logic [N-1:0][ACC_W-1:0] acc = '0;
  logic [ACC_W-1:0]        q_bias = '0;
  logic [M_W-1:0]          m = '0;
  logic [S_W-1:0]          s = '0;
  logic                    relu_en = 1'b0, out_raw = 1'b0;
  logic                    out_valid, out_raw_d;
  logic [TW-1:0]           out_tok;
  logic [N-1:0][7:0]       q_int8;
  logic [N-1:0][ACC_W-1:0] v_raw;

  gos_requant #(.TW(TW)) dut (.*);

  always #2.5 clk = ~clk;

  // ------------------------------------------------------------------ expected items
  typedef struct {
    logic [TW-1:0]           tok;
    logic                    raw;
    logic [N-1:0][7:0]       q;
    logic [N-1:0][ACC_W-1:0] v;
    longint                  cyc;
  } item_t;
  item_t exp_q[$];

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  longint n_checks = 0, n_items = 0;
  longint n_vec_rows[2] = '{0, 0}, n_vec_lane[2] = '{0, 0}, n_nt[2] = '{0, 0};
  longint n_rnd_lane = 0, n_tie_lane = 0, n_s0_lane = 0, n_model_x = 0;

  task automatic fail(input string msg);
    $display("TEST FAILED %s (cycle %0d)", msg, cyc);
    $fatal(1, "tb_gos_requant: %s", msg);
  endtask

  // ------------------------------------------------------------------ reference model
  function automatic logic [7:0] ref_q(input longint v, input logic [31:0] mm,
                                      input int ss, input bit relu);
    logic signed [127:0] p, q, vv, mw;
    if (ss == 0) return 8'h00;
    vv = 128'(v);                         // sign-extended
    mw = $signed({96'd0, mm});
    p  = vv * mw;                         // exact (|p| < 2^58 << 2^127)
    q  = (p + (128'sd1 <<< (ss - 1)) - 128'sd1 + ((p >>> ss) & 128'sd1)) >>> ss;
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    if (relu && q < 0) q = 0;
    return q[7:0];
  endfunction

  // |v| < 2^25 bound (host proof, V_MUL_W = 26)
  localparam longint VLIM = (64'sd1 <<< (V_MUL_W - 1)) - 1;

  function automatic longint clampv(input longint v);
    if (v >  VLIM) return  VLIM;
    if (v < -VLIM) return -VLIM;
    return v;
  endfunction

  function automatic longint rnd_range(input longint lo, input longint hi);  // inclusive
    longint unsigned span, r;
    span = longint'(hi - lo) + 1;
    r    = {$urandom(), $urandom()};
    return lo + longint'(r % span);
  endfunction

  // exact-region half width: |v·m/2^s| <= ~131
  function automatic longint exact_r(input logic [31:0] mm, input int ss);
    logic [127:0] r;
    if (mm == 0) return 1000;
    r = ((128'd131 << ss) / mm) + 1;
    if (r > VLIM) return VLIM;
    return longint'(r);
  endfunction

  // TB-generated lane v for given channel params
  function automatic longint gen_v(input longint vrow, input logic [31:0] mm, input int ss);
    int mode = $urandom_range(0, 3);
    longint r;
    case (mode)
      0: begin  // near the row's v
        r = rnd_range(-(64'sd1 <<< $urandom_range(0, 12)), (64'sd1 <<< $urandom_range(0, 12)));
        return clampv(vrow + r);
      end
      1, 2: begin
        r = exact_r(mm, ss);
        return rnd_range(-r, r);
      end
      default: return rnd_range(-VLIM, VLIM);
    endcase
  endfunction

  // ------------------------------------------------------------------ drive helpers
  task automatic idle_gap();
    int g;
    if ($urandom_range(0, 99) < 15) begin
      g = $urandom_range(1, 3);
      repeat (g) begin
        @(negedge clk);
        check_out();
        in_valid = 1'b0;
        in_tok   = TW'($urandom());
        for (int l = 0; l < N; l++) acc[l] = $urandom();
        q_bias = $urandom(); m = $urandom(); s = $urandom(); relu_en = $urandom(); out_raw = $urandom();
      end
    end
  endtask

  // drive one valid cycle; lane values v[l], shared params; expected q given
  task automatic drive(input longint vl[N], input logic [31:0] qb, input logic [31:0] mm,
                       input logic [5:0] ss, input bit relu, input bit raw,
                       input logic [N-1:0][7:0] qexp);
    item_t it;
    idle_gap();
    @(negedge clk);
    check_out();
    in_valid = 1'b1;
    in_tok   = TW'($urandom());
    q_bias = qb; m = mm; s = ss; relu_en = relu; out_raw = raw;
    for (int l = 0; l < N; l++) begin
      acc[l]  = 32'(vl[l]) - qb;          // acc = v - q_bias (mod 2^32)
      it.v[l] = 32'(vl[l]);
    end
    it.tok = in_tok; it.raw = raw; it.q = qexp; it.cyc = cyc;
    exp_q.push_back(it);
  endtask

  // ------------------------------------------------------------------ output checker
  task automatic check_out();
    item_t it;
    if (out_valid !== 1'b1) begin
      if (out_valid !== 1'b0) fail("out_valid is X");
      return;
    end
    if (exp_q.size() == 0) fail("out_valid with no pending item");
    it = exp_q.pop_front();
    if (cyc - it.cyc != L_RQ) fail($sformatf("latency %0d != L_RQ %0d", cyc - it.cyc, L_RQ));
    if (out_tok !== it.tok) fail($sformatf("tok %h != %h", out_tok, it.tok));
    if (out_raw_d !== it.raw) fail("out_raw_d mismatch");
    n_checks += 3;
    for (int l = 0; l < N; l++) begin
      if (q_int8[l] !== it.q[l])
        fail($sformatf("lane %0d q %0d != exp %0d (v=%0d)", l, $signed(q_int8[l]),
                       $signed(it.q[l]), $signed(it.v[l])));
      if (v_raw[l] !== it.v[l])
        fail($sformatf("lane %0d v_raw %0d != exp %0d", l, $signed(v_raw[l]), $signed(it.v[l])));
      n_checks += 2;
    end
    n_items++;
  endtask

  // ------------------------------------------------------------------ vector part
  localparam int VMAX = 131072;         // >= rows per file (100,004)
  logic [159:0] vmem [VMAX];
  task automatic run_net(input int ni, input string net, input string dir);
    int nrows;
    logic [159:0] row;
    string path;
    bit got[6];
    path = {dir, "/unit/requant/", net, "_requant.hex"};
    for (int i = 0; i < VMAX; i++) vmem[i] = 'x;
    $readmemh(path, vmem);
    nrows = 0;
    while (nrows < VMAX && !$isunknown(vmem[nrows])) begin
      row = vmem[nrows];
      begin
        logic [31:0] racc, qb, mm, rv;
        logic [5:0]  ss;
        logic        nt;
        logic [7:0]  fq, fqr, idx;
        longint      v, vl[N];
        int          lane;
        logic [N-1:0][7:0] qe;
        racc = row[31:0];   qb = row[63:32];   mm = row[95:64];  rv = row[127:96];
        ss   = row[133:128]; nt = row[134];
        fq   = row[143:136]; fqr = row[151:144]; idx = row[159:152];
        v    = longint'($signed(rv));
        if (racc + qb != rv) fail($sformatf("%s row %0d: acc + q_bias != v in file", net, nrows));
        if (v > VLIM || v < -VLIM) fail($sformatf("%s row %0d: |v| >= 2^25", net, nrows));
        if (ss == 0) fail($sformatf("%s row %0d: s = 0", net, nrows));
        if (ref_q(v, mm, ss, 0) !== fq || ref_q(v, mm, ss, 1) !== fqr) begin
          n_model_x++;
          fail($sformatf("%s row %0d: TB model disagrees with file", net, nrows));
        end
        if (nt) begin
          n_nt[ni]++;
          if (ni == 0) begin
            if (idx == 104 && v ==  55930) got[0] = 1;
            if (idx == 104 && v == -55930) got[1] = 1;
            if (idx == 196 && v ==  31044) got[2] = 1;
            if (idx == 196 && v == -31044) got[3] = 1;
          end else begin
            if (idx == 121 && v ==  36475) got[4] = 1;
            if (idx == 121 && v == -36475) got[5] = 1;
          end
        end
        lane = nrows % N;
        for (int r = 0; r < 2; r++) begin
          for (int l = 0; l < N; l++) begin
            vl[l] = (l == lane) ? v : gen_v(v, mm, ss);
            qe[l] = ref_q(vl[l], mm, ss, r);
          end
          qe[lane] = r ? fqr : fq;          // file value for the vector lane
          drive(vl, qb, mm, ss, r, 1'b0, qe);
          n_vec_lane[ni]++;
        end
      end
      nrows++;
    end
    n_vec_rows[ni] = nrows;
    if (nrows < 100000) fail($sformatf("%s: only %0d rows", net, nrows));
    if (ni == 0 && !(got[0] && got[1] && got[2] && got[3]))
      fail("lenet5 near-ties conv5 ch82 +-55930 / fc1 ch54 +-31044 missing");
    if (ni == 1 && !(got[4] && got[5]))
      fail("cifar10 near-tie conv3 ch57 +-36475 missing");
    $display("[tb] %s: %0d vector rows x2 (relu 0/1) driven, %0d near-tie rows", net, nrows, n_nt[ni]);
  endtask

  // ------------------------------------------------------------------ random part
  task automatic run_random();
    for (int c = 0; c < N_RND; c++) begin
      logic [31:0] mm, qb;
      int ss, kind;
      bit relu;
      longint vl[N];
      logic [N-1:0][7:0] qe;
      kind = $urandom_range(0, 99);
      relu = $urandom_range(0, 1);
      qb   = 32'(rnd_range(-(64'sd1 <<< 24), (64'sd1 <<< 24)));
      if (kind < 2) begin                                   // s = 0 (final-layer style)
        ss = 0;
        mm = ($urandom_range(0, 1)) ? 32'd0 : $urandom();
        for (int l = 0; l < N; l++) vl[l] = rnd_range(-VLIM, VLIM);
        n_s0_lane += N;
      end else if (kind < 35) begin                         // forced exact ties
        int lo, hi, b, a;
        logic [31:0] mo;
        ss = $urandom_range(1, 56);
        lo = (ss - 25 > 0) ? ss - 25 : 0;
        hi = (ss - 1 < 31) ? ss - 1 : 31;
        b  = ($urandom_range(0, 1) && hi >= 24 && lo <= 24) ? $urandom_range((lo > 24) ? lo : 24, hi)
                                                            : $urandom_range(lo, hi);
        a  = ss - 1 - b;
        if (b == 31) mo = 32'd1;
        else         mo = 32'(rnd_range(64'sd1 <<< (31 - b), (64'sd1 <<< (32 - b)) - 1)) | 32'd1;
        mm = mo << b;
        for (int l = 0; l < N; l++) begin
          longint umax, u;
          umax = ((64'sd1 <<< (25 - a)) - 1);             // |u|·2^a < 2^25
          if ($urandom_range(0, 1)) begin                 // small |q|: |u·mo| <= ~300
            longint lim = 300 / longint'(mo);
            if (lim < 1) lim = 1;
            if (lim > umax) lim = umax;
            u = rnd_range(0, lim);
          end else
            u = rnd_range(0, umax);
          u = u | 1;
          if (u > umax) u = umax;                          // umax is odd
          if ($urandom_range(0, 1)) u = -u;
          vl[l] = u <<< a;
          if ($urandom_range(0, 7) == 0) vl[l] = clampv(vl[l] + (($urandom_range(0, 1)) ? 1 : -1));
          else n_tie_lane++;
        end
      end else begin                                        // general random
        mm = 32'h8000_0000 | $urandom();
        ss = $urandom_range(1, 63);
        for (int l = 0; l < N; l++) vl[l] = gen_v(0, mm, ss);
      end
      for (int l = 0; l < N; l++) qe[l] = ref_q(vl[l], mm, ss, relu);
      drive(vl, qb, mm, 6'(ss), relu, (ss == 0), qe);
      n_rnd_lane += N;
    end
  endtask

  // ------------------------------------------------------------------ main
  initial begin
    string dir;
    process::self().srandom(20260924);
    if (!$value$plusargs("VEC_DIR=%s", dir)) dir = "../../../vectors/generated";
    rst = 1'b1;
    repeat (4) @(negedge clk);
    if (out_valid !== 1'b0) fail("out_valid not 0 in reset");
    rst = 1'b0;

    run_net(0, "lenet5", dir);
    run_net(1, "cifar10", dir);
    run_random();

    @(negedge clk);
    check_out();
    in_valid = 1'b0;
    repeat (L_RQ + 2) begin
      @(negedge clk);
      check_out();
    end
    if (exp_q.size() != 0) fail($sformatf("%0d items never came out", exp_q.size()));

    $display("[tb] items=%0d | vector rows: lenet5 %0d, cifar10 %0d (each x2: relu 0/1; lane cycles %0d/%0d)",
             n_items, n_vec_rows[0], n_vec_rows[1], n_vec_lane[0], n_vec_lane[1]);
    $display("[tb] near-tie rows checked: lenet5 %0d, cifar10 %0d (x2 relu); required near-ties present",
             n_nt[0], n_nt[1]);
    $display("[tb] random lanes %0d (exact-tie lanes %0d, s=0 lanes %0d); vector-cycle filler lanes %0d",
             n_rnd_lane, n_tie_lane, n_s0_lane, 7 * (n_vec_lane[0] + n_vec_lane[1]));
    $display("TEST PASSED checks=%0d", n_checks);
    $finish;
  end

endmodule
