include("mnist.jl")

x, t = get_data()
batch_size = 100
accuracy_cnt = 0

for i in 1:batch_size:size(x,2)
    x_batch = x[i:i+batch_size-1,:]
    y_batch = predict_batch(network,x_batch)
    p=mapslices(argmax, y_batch, dims=2).-1
    global accuracy_cnt += sum(p .== t[i:i+batch_size-1])
end

print(accuracy_cnt/size(x,2))