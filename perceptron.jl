function AND(x1, x2)
    x = [x1, x2]
    w = [0.5, 0.5]
    b = -0.7
    tmp = x'*w + b
    if tmp <= 0
        return 0
    else
        return 1
    end
end

function NAND(x1, x2)
    x = [x1, x2]
    w = [-0.5, -0.5]
    b = 0.7
    tmp = x'*w + b
    if tmp <= 0
        return 0
    else
        return 1
    end
end

function OR(x1, x2)
    x = [x1, x2]
    w = [0.5, 0.5]
    b = -0.2
    tmp = x'*w + b
    if tmp <= 0
        return 0
    else
        return 1
    end
end

function XOR(x1, x2)
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
end
