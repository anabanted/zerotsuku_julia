using Colors
using Plots
using Pickle
using PyCall

include("load_mnist.jl")
include("NN.jl")

pickle = pyimport("pickle")
np = pyimport("numpy")


function img_show(img)
    draw_img = Gray.(img./255)
    plot(draw_img)
end

function get_data()
    x_train, t_train, x_test, t_test = load_mnist(normalize=false, flatten=true, one_hot_label=false)
    return x_test, t_test
end

load_pickle(filename) = begin
    @pywith pybuiltin("open")(filename, "rb") as f begin
        return pickle.load(f)
    end
end

function init_network()
    network = load_pickle("sample_weight.pkl")
end

function predict(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = x'*W1 + b1'
    z1 = sigmoid.(a1)
    a2 = z1*W2 + b2'
    z2 = sigmoid.(a2)
    a3 = z2*W3 + b3'
    y = softmax(a3)
    
    return y
end

function predict_batch(network, x)
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    
    a1 = x*W1 .+ b1'
    z1 = sigmoid.(a1)
    a2 = z1*W2 .+ b2'
    z2 = sigmoid.(a2)
    a3 = z2*W3 .+ b3'
    y = softmax(a3)
    
    return y
end