FUNCTION_BLOCK CNN1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR CONSTANT
    weights_0 : ARRAY[0..5] OF REAL := [-0.757294893, 1.60322511, -0.810065687, -0.0401188135, -0.0341094136, 2.28848505];
    bias_0 : ARRAY[0..5] OF REAL := [0, -0.327996045, 0, 0, 0, -0.425784826];


    weights_2 : ARRAY[0..95] OF REAL := [0.387440026, -0.331604958, -0.485271782, -0.178175241, -0.261189312, -0.126141995, -0.41867274, 0.10701102, -0.504957795, -0.0640092492, -0.506339848, -0.0112774968, -0.248149365, 0.370121658, 0.102361083, 0.514361918, 2.27920318, -2.03401303, 2.57838869, 0.345396936, 1.7697078, 2.31489539, 1.82641065, 0.116435051, 2.60688972, -0.141314536, 0.0899649858, -0.115778536, -0.130346328, -2.0967865, 1.73134959, 1.73764074, -0.125548214, -0.418661654, 0.42383492, 0.245940804, -0.231652409, -0.521428883, -0.369046271, 0.500959337, 0.39457792, -0.498766184, 0.505615175, 0.00158774853, -0.358177781, 0.0681350231, 0.211269855, 0.317975223, 0.470600903, -0.22380206, -0.221884102, 0.369123101, -0.365577281, 0.470880032, 0.0877029896, -0.22456941, 0.125117779, 0.0604217052, 0.403043449, 0.0515841246, 0.360347748, 0.176390409, 0.499794424, 0.4554739, 0.357078254, -0.20554477, -0.338528842, -0.346329391, -0.316419125, -0.192169428, 0.340339243, -0.170743108, 0.503388941, 0.431303263, 0.182139158, 0.344671071, 0.0957750082, -0.455443144, 0.516620576, -0.12648204, 2.20542169, -1.42068136, 1.77867258, -0.279205918, 1.31937492, 2.40089417, 1.49984801, -0.381873012, 2.37344623, -0.157825798, -0.164687812, -0.438219547, -0.030656606, -0.809033334, 1.04967833, 1.52932751];
    bias_2 : ARRAY[0..15] OF REAL := [-0.00126687379, 1.46715438, -0.0669283643, 0, -0.544165969, -0.00420819549, -0.593534648, 0, -0.00139115693, 0, 0, 0, 0, 1.40235019, -0.471719682, -0.589310586];

    weights_3 : ARRAY[0..47] OF REAL := [-2.44326067, 1.56872427, 1.00637841, 2.21994734, -0.0721509904, -2.70681095, -2.57062817, 0.8563025, 1.35775805, 0.436492383, -0.214310527, 0.435115576, -0.357676566, -2.57688642, 2.20074344, -1.96935773, 1.55129766, 1.27391887, -0.475576401, -2.55197287, 1.79706705, -0.406806469, -0.373194754, 0.021515131, -1.98918331, 1.86888134, 1.35972965, 0.224319875, -0.156571329, -0.38576901, -0.264996111, -0.172905713, 0.307211518, 0.210194111, -0.266009122, -0.164967567, -0.275477469, 0.512401879, 0.539515197, 1.5903399, -0.0624566786, -2.02082109, -1.80255735, -3.25323439, 2.98463917, -0.251099586, -2.6509831, 1.72883391];
    bias_3 : ARRAY[0..2] OF REAL := [0.74891752, -0.036865145, -0.95004487];

END_VAR

VAR
    (* Merged variables from all regions *)
    buffer_0 : ARRAY[0..5] OF REAL;
    buffer_1 : ARRAY[0..15] OF REAL;

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
(* Layer 0: Conv2D  in=1x1x1  out=6x1x1  kernel=1x1  stride=1x1  pad=(0, 0, 0, 0)  group=1 *)
FOR oc := 0 TO 5 DO
    FOR oh := 0 TO 0 DO
        FOR ow := 0 TO 0 DO
            sum := bias_0[oc];
            FOR ic := 0 TO 0 DO
                FOR kh := 0 TO 0 DO
                    FOR kw := 0 TO 0 DO
                        ih := oh * 1 - 0 + kh;
                        iw := ow * 1 - 0 + kw;
                        sum := sum + input_data * weights_0[oc * 1 + ic * 1 + kh * 1 + kw];
                    END_FOR;
                END_FOR;
            END_FOR;
            buffer_1[oc * 1 + oh * 1 + ow] := sum;
        END_FOR;
    END_FOR;
END_FOR;

(* Layer 1: Activation (RELU) *)
FOR i := 0 TO 5 DO
    buffer_0[i] := MAX(buffer_1[i], 0.0);
END_FOR;

(* Layer 2: Fused Linear + RELU *)
FOR j := 0 TO 15 DO
    sum := 0.0;
    FOR i := 0 TO 5 DO
        sum := sum + buffer_0[i] * weights_2[i * 16 + j];
    END_FOR;
buffer_1[j] := MAX(sum + bias_2[j], 0.0);
END_FOR;

(* Layer 3: Fused Linear + SOFTMAX *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 15 DO
        sum := sum + buffer_1[i] * weights_3[i * 3 + j];
    END_FOR;
output_data[j] := sum + bias_3[j];
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
