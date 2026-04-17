FUNCTION_BLOCK MLP1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    CONSTANT weights_0 : ARRAY[0..7] OF REAL := [-0.39633444, -0.548940063, -0.0548924804, 2.65775609, -0.186861873, 1.99443269, -0.4275029, 2.42555642];
    CONSTANT bias_0 : ARRAY[0..7] OF REAL := [0, 0, 0, -0.386175603, 1.51318884, -0.335069925, 0, -0.41231662];
    CONSTANT weights_1 : ARRAY[0..63] OF REAL := [-0.249718159, 0.355077267, 0.582239211, 0.299669743, 0.560839117, -0.478885651, -0.387439847, 0.0731658936, 0.468102634, 0.0799882412, -0.327028006, 0.219414055, -0.399431229, -0.0365031362, -0.260698169, -0.198557079, -0.294705153, -0.244047046, 0.125463724, 0.588593543, 0.190051973, -0.535617709, -0.0162392259, 0.530301034, 1.90118039, 1.77590561, -0.333346367, -0.0687134862, -0.151702613, -0.455085605, -0.929370284, -1.71230757, 0.0891756192, 0.0794340521, 0.0402879156, -0.0160570145, -0.381342113, -0.337033749, 1.15654266, 1.46721661, 2.8483572, 2.19387126, -0.00236231089, -0.593520105, -0.337822169, -0.402935982, -1.80942917, -2.64788198, -0.0498700142, 0.161640584, -0.224173844, -0.441094339, 0.0303041339, 0.452861011, -0.248058438, -0.0450696349, 2.57059598, 1.8905555, -0.60645926, -0.0833607316, 0.233341992, -0.298924446, -2.09441996, -2.33421278];
    CONSTANT bias_1 : ARRAY[0..7] OF REAL := [-0.133272439, -0.397205234, -0.0605889186, 0, 0, 0, 1.00933897, 1.0929544];
    CONSTANT weights_2 : ARRAY[0..23] OF REAL := [-2.8537457, 0.653815091, 1.25764716, -1.93262327, 0.2645832, 1.83192003, -0.624833703, 0.0545924753, 0.68312341, -0.0164062381, -0.21331352, -0.586643577, -0.673535287, -0.466759026, -0.15310818, -0.313170284, -0.110788405, 0.485940039, 1.39489615, 0.230346501, -2.96772361, 2.38400936, 0.345051408, -2.33157396];
    CONSTANT bias_2 : ARRAY[0..2] OF REAL := [0.0344888456, 0.284831911, -0.563379645];
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
