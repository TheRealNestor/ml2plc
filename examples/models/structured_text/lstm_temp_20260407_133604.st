FUNCTION_BLOCK lstm_temp_20260407_133604

    (* Model Inputs/Outputs and State Variables *)
VAR
    (* Internal variables for all regions *)
END_VAR
(* Forward pass execution *)

(* Region: r0 [ACYCLIC] *)
(* Acyclic Region r0 *)

(* Layer 0: Transpose perm=(0, 1)  (20, 1) -> (20, 1) *)
FOR t0_d0 := 0 TO 19 DO
    FOR t0_d1 := 0 TO 0 DO
        buffer_2[t0_d0 + t0_d1] := input_data[t0_d0 + t0_d1];
    END_FOR;
END_FOR;

(* Layer 1: Squeeze *)
FOR i := 0 TO 31 DO
    buffer_1[i] := input_data[i];
END_FOR;

(* Layer 2: LSTM (hidden_size=32, input_size=1) *)
(* Single timestep recurrent computation with 4 gates *)

(* Input Gate: i_t = sigmoid(W_i @ x + R_i @ h_{t-1} + b_i) *)
FOR j := 0 TO 31 DO
    sum := 0.0;
    FOR i := 0 TO 0 DO
        sum := sum + buffer_2[i] * weights_2_i[i * 32 + j];
    END_FOR;
    FOR i := 0 TO 31 DO
        sum := sum + h_state_in_2[i] * recurrent_2_i[i * 32 + j];
    END_FOR;
    sum := sum + bias_2_i[j];
    i_gate_2[j] := 1.0 / (1.0 + EXP(-sum));  (* sigmoid *)
END_FOR;

(* Forget Gate: f_t = sigmoid(W_f @ x + R_f @ h_{t-1} + b_f) *)
FOR j := 0 TO 31 DO
    sum := 0.0;
    FOR i := 0 TO 0 DO
        sum := sum + buffer_2[i] * weights_2_f[i * 32 + j];
    END_FOR;
    FOR i := 0 TO 31 DO
        sum := sum + h_state_in_2[i] * recurrent_2_f[i * 32 + j];
    END_FOR;
    sum := sum + bias_2_f[j];
    f_gate_2[j] := 1.0 / (1.0 + EXP(-sum));  (* sigmoid *)
END_FOR;

(* Cell Gate (Candidate): g_t = tanh(W_g @ x + R_g @ h_{t-1} + b_g) *)
FOR j := 0 TO 31 DO
    sum := 0.0;
    FOR i := 0 TO 0 DO
        sum := sum + buffer_2[i] * weights_2_g[i * 32 + j];
    END_FOR;
    FOR i := 0 TO 31 DO
        sum := sum + h_state_in_2[i] * recurrent_2_g[i * 32 + j];
    END_FOR;
    sum := sum + bias_2_g[j];
    exp_val := EXP(2.0 * sum);
    g_gate_2[j] := (exp_val - 1.0) / (exp_val + 1.0);  (* tanh *)
END_FOR;

(* Output Gate: o_t = sigmoid(W_o @ x + R_o @ h_{t-1} + b_o) *)
FOR j := 0 TO 31 DO
    sum := 0.0;
    FOR i := 0 TO 0 DO
        sum := sum + buffer_2[i] * weights_2_o[i * 32 + j];
    END_FOR;
    FOR i := 0 TO 31 DO
        sum := sum + h_state_in_2[i] * recurrent_2_o[i * 32 + j];
    END_FOR;
    sum := sum + bias_2_o[j];
    o_gate_2[j] := 1.0 / (1.0 + EXP(-sum));  (* sigmoid *)
END_FOR;

(* Cell State Update: c_t = f_t * c_{t-1} + i_t * g_t *)
FOR j := 0 TO 31 DO
    c_state_out_2[j] := f_gate_2[j] * c_state_in_2[j] + i_gate_2[j] * g_gate_2[j];
END_FOR;

(* Output Hidden State: h_t = o_t * tanh(c_t) *)
FOR j := 0 TO 31 DO
    exp_val := EXP(2.0 * c_state_out_2[j]);
    c_tanh_2[j] := (exp_val - 1.0) / (exp_val + 1.0);
    buffer_0[j] := o_gate_2[j] * c_tanh_2[j];
END_FOR;

(* Layer 3: Fused Linear + RELU *)
FOR j := 0 TO 15 DO
    sum := 0.0;
    FOR i := 0 TO 31 DO
        sum := sum + buffer_1[i] * weights_3[i * 16 + j];
    END_FOR;
    buffer_2[j] := MAX(sum + bias_3[j], 0.0);
END_FOR;

(* Layer 4: Squeeze *)
FOR i := 0 TO 639 DO
    output_data[i] := buffer_0[i];
END_FOR;

(* Layer 5: Fused Linear + SOFTMAX *)
FOR j := 0 TO 2 DO
    sum := 0.0;
    FOR i := 0 TO 15 DO
        sum := sum + buffer_2[i] * weights_5[i * 3 + j];
    END_FOR;
    output_data[j] := sum + bias_5[j];
END_FOR;

max_val := output_data[0];
FOR i := 1 TO 2 DO
    IF output_data[i] > max_val THEN
        max_val := output_data[i];
    END_IF;
END_FOR;

exp_sum := 0.0;
FOR i := 0 TO 2 DO
    output_data[i] := EXP(output_data[i] - max_val);
    exp_sum := exp_sum + output_data[i];
END_FOR;

FOR i := 0 TO 2 DO
    output_data[i] := output_data[i] / exp_sum;
END_FOR;


END_FUNCTION_BLOCK
