FUNCTION_BLOCK CNN1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : ARRAY[0..0] OF REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR
    CONSTANT weights_0 : ARRAY[0..5] OF REAL := [-0.598121881, 2.32422185, -0.270557165, -0.407380521, -0.521352589, 2.15304375];
    CONSTANT bias_0 : ARRAY[0..5] OF REAL := [0, -0.434294552, 0, 0, 0, -0.43597427];
    CONSTANT weights_2 : ARRAY[0..95] OF REAL := [-0.333467633, -0.41049403, -0.0482003391, 0.469121337, 0.426341653, -0.286234021, -0.282458127, 0.00599837303, -0.121161699, -0.300314605, 0.0491855741, 0.357676864, 0.227587163, -0.198910385, -0.182846606, -0.0328884423, -0.0994983315, 1.97408879, 1.01767349, 2.38331532, -0.436812967, -0.657083631, 1.38934743, 1.58582067, -0.178655595, -0.252660364, -0.477673173, 0.804847538, 2.14524102, 0.819824636, 2.04465675, -1.81944013, -0.213972479, 0.464753866, -0.422858626, -0.501581609, 0.222119808, -0.384456754, 0.389178813, -0.374694884, -0.119004339, 0.0463869572, -0.0880898535, -0.32528162, 0.0845478177, -0.518456459, 0.272620201, -0.0342045128, -0.0850790739, -0.183166474, 0.202489674, 0.0944411159, -0.256414235, 0.00192826986, -0.119364172, 0.500171602, 0.492882431, 0.521977723, 0.256548464, 0.243987143, 0.267252982, 0.0935279727, -0.147054434, 0.249463081, 0.463899612, 0.21559757, -0.0998596549, -0.405632794, -0.0965154469, 0.465933263, 0.186893046, 0.270634532, 0.0107021332, 0.452107251, 0.284197509, -0.386763662, -0.123662382, 0.400962055, -0.0607939065, -0.0441588461, -0.0207372904, 2.02401829, 1.70029426, 2.73039007, -0.275693476, -1.70780933, 1.62482488, 0.991947651, -0.155292034, -0.490615606, 0.0549013019, 1.56200385, 2.34798193, 1.01525831, 1.83524668, -2.44287014];
    CONSTANT bias_2 : ARRAY[0..15] OF REAL := [0, -0.00323987985, -0.545170486, -0.0067349337, 0, 1.46903372, -0.617733061, -0.546088815, 0, 0, 0, -0.469247669, -0.00387184462, -0.373564959, -0.0162188746, 1.46983624];
    CONSTANT weights_3 : ARRAY[0..47] OF REAL := [-0.0623795986, -0.125149995, -0.251098394, -2.09429932, 1.5382247, 1.07247424, -0.157979235, -2.32853031, 1.98592734, -2.44551206, 0.581950784, 0.360890388, 0.181147635, -0.443369269, -0.335153878, 1.36539197, 0.156253964, -2.35229373, -0.758998334, -2.77431679, 1.35183728, -0.720050752, -2.84092832, 2.43330812, 0.523565471, 0.174443662, 0.214798629, 0.416738153, -0.374915421, -0.484684974, 0.441833556, 0.359905541, 0.217649996, -0.415302545, -2.86921477, 2.50061584, -1.86696982, 1.17750669, 0.878900468, 0.138796464, -1.63971829, 1.15392041, -2.1682179, 1.19688225, 1.05973184, 2.31370115, -0.720425606, -2.65744376];
    CONSTANT bias_3 : ARRAY[0..2] OF REAL := [0.680391073, 0.188020289, -0.979192913];
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
                        sum := sum + input_data[ic * 1 + ih * 1 + iw] * weights_0[oc * 1 + ic * 1 + kh * 1 + kw];
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
