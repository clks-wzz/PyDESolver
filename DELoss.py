#lucky
def loss_test(input_list):
    input_list = list(input_list)
    assert(len(input_list)==3)
    a, b, c = input_list
    loss = (a - 1.0) ** 2 + (b - 2.0) ** 2 + (c - 3.0) ** 2
    return loss