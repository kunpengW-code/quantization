def create_linear_layer(
    quant_method,
    input_size=128,
    output_size=256,
    params_dtype=torch.bfloat16,
):
    layer = nn.Module()
    weight_dict = quant_method.get_weight(input_size, output_size, params_dtype)
    for weight_name, weight_param in weight_dict.items():
        param = torch.nn.Parameter(weight_param.npu(), requires_grad=False)
        layer.register_parameter(weight_name, param)

    pertensor_dict = quant_method.get_pertensor_param(params_dtype)
    for pertensor_name, pertensor_param in pertensor_dict.items():
        param = torch.nn.Parameter(pertensor_param.npu(), requires_grad=False)
        layer.register_parameter(pertensor_name, param)

    perchannel_dict = quant_method.get_perchannel_param(output_size, params_dtype)
    for perchannel_name, perchannel_param in perchannel_dict.items():
        param = torch.nn.Parameter(perchannel_param.npu(), requires_grad=False)
        layer.register_parameter(perchannel_name, param)

    layer_type = "row" # if isinstance(layer, RowParallelLinear) else "others"
    pergroup_dict = quant_method.get_pergroup_param(input_size, output_size, params_dtype, layer_type=layer_type)
    for pergroup_name, pergroup_param in pergroup_dict.items():
        param = torch.nn.Parameter(pergroup_param.npu(), requires_grad=False)
        layer.register_parameter(pergroup_name, param)
