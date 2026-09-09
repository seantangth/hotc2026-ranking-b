def calculate_overlaps(a_list, b_list):
    out = []
    for a, b in zip(a_list, b_list):
        ix = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
        iy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
        inter = ix * iy
        union = a.width * a.height + b.width * b.height - inter
        out.append(inter / union if union > 0 else 0.0)
    return out
