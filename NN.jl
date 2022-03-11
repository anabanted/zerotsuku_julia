function step_function(x)
    y = Int(x.>0)
    return y
end

function sigmoid(x)
    return 1/(1+exp(-x))
end

function relu(x)
    return (max(0,x))    
end

function identity_function(x)
    return x
end

function init_network()
    network = Dict()
    network["W1"] = [0.1 0.3 0.5; 0.2 0.4 0.6]
    network["b1"] = [0.1 0.2 0.3]
    network["W2"] = [0.1 0.4; 0.2 0.5; 0.3 0.6]
    network["b2"] = [0.1 0.2]
    network["W3"] = [0.1 0.3; 0.2 0.4]
    network["b3"] = [0.1 0.2]
    
    return network
end

function forward(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = x*W1 + b1
    z1 = sigmoid.(a1)
    a2 = z1*W2 + b2
    z2 = sigmoid.(a2)
    a3 = z2*W3 + b3
    y = identity_function.(a3)
    
    return y
end

function softmax(a)
    c = maximum(a)
    exp_a = exp.(a .- c)
    sum_exp_a = sum(exp_a)
    y = exp_a ./ sum_exp_a

    return y
end