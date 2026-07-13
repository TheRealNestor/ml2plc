FUNCTION_BLOCK MLP1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR CONSTANT
    weights_0 : ARRAY[0..7] OF REAL := [-0.359626085, -0.243904531, -0.401037216, 0.64431119, 0.232034206, 0.792446733, 0.228110984, -0.0573521256];
    bias_0 : ARRAY[0..7] OF REAL := [0, 0, 0, -0.0624486357, -0.00978165679, -0.0783689246, 0.0478095934, 0];

    weights_1 : ARRAY[0..63] OF REAL := [-0.448344469, -0.406341732, 0.181895137, -0.3638134, -0.402484953, -0.0546350479, 0.00203305483, 0.474552453, -0.581456602, 0.274176121, -0.485772967, -0.549738646, -0.342836112, -0.27996707, -0.441847682, 0.0135117769, -0.436782479, -0.103253484, -0.101985157, 0.0595596433, -0.510617077, 0.357503235, 0.421454012, 0.0561642647, -0.275652379, -0.515787542, -0.507303357, -0.351691812, -0.367938966, 0.154912531, 0.2581532, 0.449660361, -0.226280138, -0.57086134, 0.0189548358, 0.189382374, 0.0692978427, 0.00651363889, 0.701111376, -0.217280686, 0.29560107, 0.0987591743, -0.323079258, -0.430943608, -0.615357518, 0.158748299, 0.669972181, -0.34587732, -0.11576958, -0.344211757, 0.332252741, -0.488115549, 0.198065117, -0.601275384, 0.645542502, -0.256326318, 0.29974103, -0.341316849, 0.554455578, 0.0207368135, -0.33459115, 0.232482135, 0.232025743, -0.152884334];
    bias_1 : ARRAY[0..7] OF REAL := [-0.0174692422, 0, -0.0275537241, 0, 0.121660262, -0.0528285801, -0.0430694446, 0];

    weights_2 : ARRAY[0..23] OF REAL := [-0.183575451, 0.00586040644, -0.552747905, -0.649368286, -0.560685456, 0.373448908, -0.638512611, 0.0955652669, 0.0739731565, 0.343407094, 0.305277169, -0.563019454, 0.584188223, -0.777326345, -0.318548292, 0.535584629, 0.257676065, -0.0411909297, -0.068027094, -0.119331792, 0.615767598, 0.418245614, 0.0741652846, -0.445219398];
    bias_2 : ARRAY[0..2] OF REAL := [0.00981868524, 0.0205980819, -0.0361016765];

END_VAR

VAR
    (* Merged variables from all regions *)
    buffer_0 : ARRAY[0..7] OF REAL;
    buffer_1 : ARRAY[0..7] OF REAL;

    (* Temporary computation variables *)
    i : DINT;
    j : DINT;
    t : DINT;
    sum : REAL;
    (* Computation helpers *)
    max_val : REAL;
    exp_val : REAL;
    exp_sum : REAL;
    (* Spatial loop variables *)
    oc : DINT;
    oh : DINT;
    ow : DINT;
    ic : DINT;
    kh : DINT;
    kw : DINT;
    ih : DINT;
    iw : DINT;
END_VAR

(* Forward pass execution *)

(* Region: r0 [ACYCLIC] *)
(* Acyclic Region r0 *)
(* Layer 0: Fused Linear + RELU *)
FOR j := 0 TO 7 DO
    sum := 0.0;
    FOR i := 0 TO 0 DO
        sum := sum + input_data * weights_0[i * 8 + j];
    END_FOR;
buffer_1[j] := MAX(sum + bias_0[j], 0.0);
END_FOR;

(* Layer 1: Fused Linear + RELU *)
FOR j := 0 TO 7 DO
    sum := 0.0;
    FOR i := 0 TO 7 DO
        sum := sum + buffer_1[i] * weights_1[i * 8 + j];
    END_FOR;
buffer_0[j] := MAX(sum + bias_1[j], 0.0);
END_FOR;

(* Layer 2: Fused Linear + SOFTMAX *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 7 DO
        sum := sum + buffer_0[i] * weights_2[i * 3 + j];
    END_FOR;
output_data[j] := sum + bias_2[j];
END_FOR;

(* Softmax: find max *)
max_val := output_data[0];
FOR i := 1 TO 2 DO
    IF output_data[i] > max_val THEN
        max_val := output_data[i];
    END_IF;
END_FOR;

(* Softmax: compute exponentials and sum *)
exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(output_data[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

(* Softmax: normalize *)
FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;

END_FUNCTION_BLOCK
