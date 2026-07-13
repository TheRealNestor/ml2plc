FUNCTION_BLOCK CNN1_FB

    (* Model Inputs/Outputs *)
VAR_INPUT
    input_data : REAL;
END_VAR

VAR_OUTPUT
    output_data : ARRAY[0..2] OF REAL;
END_VAR

VAR CONSTANT
    weights_0 : ARRAY[0..5] OF REAL := [1.03629839, -0.213786244, 0.19376868, -0.522091627, 0.62388128, 0.14296706];
    bias_0 : ARRAY[0..5] OF REAL := [-0.108297229, 0, 0.00760146417, 0, -0.0474790744, 0.0884008631];


    weights_2 : ARRAY[0..95] OF REAL := [0.0768629983, -0.509010971, 0.338670313, 0.434391528, -0.409168333, 0.036785841, -0.579648674, -0.435444355, 0.604630768, -0.236359432, -0.113374859, 0.0376898572, -0.411882192, 0.383732051, -0.0465382338, 0.149891362, 0.12904954, -0.085531801, 0.332268476, 0.314284503, -0.492416531, 0.0111663342, -0.0747662783, 0.469970882, 0.492331922, -0.281263441, 0.412632644, -0.275643945, 0.473797917, -0.485291809, 0.44033289, 0.0419581532, -0.449715376, 0.0748844743, 0.691275656, 0.239018396, 0.460407138, -0.475473583, 0.419744194, -0.194104552, 0.689850092, 0.265881121, 0.405926228, -0.52667892, 0.135527194, -0.387379438, 0.127851009, -0.577873945, 0.373191833, 0.329220951, 0.0463392735, -0.0489437878, 0.362876058, 0.264890015, 0.292819977, 0.378023088, 0.297507286, -0.227719903, -0.123056501, 0.253032804, -0.434370697, 0.0781450868, -0.496926159, -0.343426704, 0.334930688, 0.32751447, 0.384838372, -0.532313466, 0.342394322, -0.0736180544, 0.14043653, 0.484879196, 0.686670899, -0.496831447, 0.0779017285, -0.168488547, 0.18528229, 0.396640331, -0.1779719, 0.131245717, 0.1077663, -0.41713801, -0.0296907611, -0.430720419, 0.0997475088, -0.211349308, 0.130485415, -0.429778278, -0.148076877, 0.299903154, -0.00105218159, 0.196733832, -0.354309142, -0.544305444, -0.237735242, -0.21331583];
    bias_2 : ARRAY[0..15] OF REAL := [-0.0436927341, 0, -0.0225529689, -0.0236954931, -0.0310794711, 0, 0.181632251, 0, -0.0430313163, 0.217456743, -0.0199339353, 0.087103337, 0, -0.032507427, 0, 0.0633436367];

    weights_3 : ARRAY[0..47] OF REAL := [0.324213713, -0.458849609, 0.316010684, 0.194880009, -0.136470884, -0.0254289508, -0.709924817, 0.0686993822, 0.73004055, 0.272423774, -0.232961118, -0.107259393, 0.216204509, 0.133011743, 0.487271249, -0.112244427, 0.253907025, 0.548123181, 0.766486824, 0.338312417, 0.0376166515, -0.0852595568, -0.554062307, 0.239705563, -0.58964473, -0.58287853, 0.27657643, 0.0132742152, -0.180403978, -0.467528492, -0.485641271, 0.520427644, -0.0898869038, 0.314900726, 0.326308012, -0.368308604, 0.466543376, 0.511979759, -0.312825501, 0.0455968529, -0.41727078, 0.0535487607, 0.554874241, 0.0828483105, -0.503180981, 0.371822089, 0.262962282, 0.0408065505];
    bias_3 : ARRAY[0..2] OF REAL := [0.0292902198, 0.0282441117, -0.0759064928];

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
