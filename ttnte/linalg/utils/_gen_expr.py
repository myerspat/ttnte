import cotengra as ctg


def gen_expr(num_cores):
    # Generate matrix-vector contraction expressions
    ex = ""
    inn = ""
    out = ""
    idx = 0
    for _ in range(num_cores):
        # Add core to expression
        chars = [ctg.get_symbol(i) for i in range(idx, idx + 4)]
        ex += "".join(chars) + ","

        # Save indices for hitting vector
        inn += chars[2]
        out += chars[1]

        idx += 3

    ex = ex[1:-2] + ","
    ex += f"{inn}->{out}"

    return ex
