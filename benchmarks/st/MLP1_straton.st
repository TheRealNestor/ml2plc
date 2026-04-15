FUNCTION_BLOCK MLP1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : ARRAY[0..0] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    CONSTANT weights_0 : ARRAY[0..7] OF REAL := [-0.618657529, -0.00120693445, -0.195586383, 1.8367641, -0.713450909, -0.470394701, -0.706411302, -0.0898849964];
    CONSTANT bias_0 : ARRAY[0..7] OF REAL := [0, 0, 0, -0.242034182, 0, 0, 0, 0];
    CONSTANT weights_1 : ARRAY[0..63] OF REAL := [0.297109783, -0.4424555, -0.413623393, -0.290249348, 0.132308125, 0.313078225, -0.487493157, 0.250414908, 0.235484362, 0.388387859, -0.295190752, -0.188208967, 0.42747885, 0.221702039, -0.176558256, 0.0120221376, 0.497074783, 0.0920943618, -0.0660725832, -0.545459628, 0.34822607, -0.221478522, -0.578834891, -0.460712194, -0.405059993, -0.381253511, 1.12712145, -0.450349629, -0.481895924, 1.69954741, -0.585079491, -0.134628654, -0.528937876, 0.219800651, 0.273241818, 0.222248971, 0.0305292606, -0.219129056, 0.191197038, 0.181811929, -0.00913119316, 0.30988735, -0.5879125, -0.0586614609, -0.306479096, -0.340859711, -0.389740527, 0.404116333, 0.295039773, 0.0847424865, 0.533310354, -0.318942487, 0.568938076, 0.0626795292, 0.344402134, -0.270115674, 0.533504903, 0.420768201, 0.395667732, -0.191096425, -0.192131132, 0.130665183, -0.477508873, -0.46032995];
    CONSTANT bias_1 : ARRAY[0..7] OF REAL := [0, 0, -0.0518381707, 0, 0, -0.204353347, 0, 0];
    CONSTANT weights_2 : ARRAY[0..23] OF REAL := [0.0885141492, 0.131608725, 0.0337964892, 0.0200446844, -0.189480245, -0.337204844, -0.821714282, -0.00229147146, 0.503815114, -0.527227759, -0.299020022, -0.511391878, 0.177334487, 0.31132561, -0.460703671, -1.68797731, -0.944721043, 0.762669921, 0.515688121, -0.577826142, 0.601135075, -0.402580202, -0.263747633, -0.414604425];
    CONSTANT bias_2 : ARRAY[0..2] OF REAL := [0.405477822, 0.17562522, -0.643234849];
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
        sum := sum + input_data[i] * weights_0[i * 8 + j];
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
