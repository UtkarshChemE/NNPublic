using DrWatson
@quickactivate "NNFitter"

using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end
using Flux #, CuArrays
using Flux: @epochs
using Printf
using Flux.Data: DataLoader
using IterTools: ncycle
using Statistics: mean, std
# using LinearAlgebra: norm, abs
using Random: shuffle
# using Plots, StatsPlots
import BSON, ROCAnalysis
# pyplot()


function main()
    segNum = parse(Int, ARGS[1])
    epochs = parse(Int, ARGS[2])

    BSON.@load datadir("xyvals", "xyvals$segNum.bson") xydata
    @info "Loaded Learning data"
    xydata = permutedims(xydata)

    # Creating normalized(X) and Y
    idx = findall(x-> x > 0.5, xydata[4,:]) #Eliminate all none solved
    X = xydata[1:3, idx]
    Y = replace(xydata[4:4, idx], 2.0=>-1) 
    μx = mean(X; dims=2)
    σx = std(X; dims=2, mean=μx, corrected=false)
    X = (X .- μx) ./ σx .* (σx .> 1e-4)  |> gpu
    
    # Chunking and creating a balanced dataset
    println("Acceptable points are ", length(idx))
    infeas_idx = shuffle(findall(x-> x < 0.0, Y[:]))
    println("Infeasible points are ", length(infeas_idx))
    feas_idx = shuffle(findall(x-> x ≥ 0.0, Y[:]))
    println("Feasible points are ", length(feas_idx))
    if length(infeas_idx)==0
        println("Classification not possible as no infeasible points are present")
        exit()
    end
    infeas_frac = floor(Int, 1000*length(infeas_idx)/length(idx))
    test_idx = [feas_idx[1:(1000-infeas_frac)]; infeas_idx[1:infeas_frac]]
    feas_idx = feas_idx[(1001-infeas_frac):end]
    infeas_idx = infeas_idx[(infeas_frac+1):end]
    weight = -ceil(Int, length(feas_idx)/length(infeas_idx))
    Y = replace(Y, -1.0 => weight) |> gpu
    nchunks = ceil(Int, (length(idx)-length(test_idx))/2^13)
    feas_chunks = Flux.chunk(feas_idx, nchunks)
    infeas_chunks = Flux.chunk(infeas_idx, nchunks)
    train_idx = reduce.(vcat, zip(feas_chunks, infeas_chunks))
    bs = maximum(length, train_idx)
    train_idx = reduce(vcat, train_idx)

    #Creating Training and testing data
    X_train = X[1:3, train_idx] |> gpu
    Y_train = Y[1:1, train_idx] |> gpu
    X_test = X[1:3, test_idx] |> gpu
    Y_test = Y[1:1, test_idx] |> gpu

    train_loader = DataLoader(X_train, Y_train; batchsize = bs)
    @info "Training data created. Training Starts"
    println("Epoch Number is 0")
    println("loss(xtrain, ytrain) = ", loss_test(X_train, Y_train), 
        " loss(xtest, ytest) = ", loss_test(X_test,Y_test))
    println("Current Accuracy = ", accuracy(X_test, Y_test))
    Flux.train!(loss, ps, train_loader, opt)
    
    for i=1:epochs
        @info "Starting epoch $i"
        Flux.train!(loss, ps, ncycle(train_loader, 5), opt)
        if i%100==0
            println("Epoch Number is $i")
            println("loss(xtrain, ytrain) = ", loss_test(X_train, Y_train), 
                " loss(xtest, ytest) = ", loss_test(X_test,Y_test))
            println("Current Accuracy = ", accuracy(X_test, Y_test))
            save_epoch(μx, σx)
            println("Should I stop? answer 1 to stop")
            if readline() == "1"
                break
            end
        end
    end
    println("Total Accuracy = ", accuracy(X, Y))
    eer = calc_eer(X, Y)
    @printf("The crossover error rate is %.3e\n", eer)
    save_params(segNum, μx, σx, eer)
end #main
    
function update_test_mod!()::Nothing
    ps_test[1] .= model.layers[2].γ .* ps[1] ./ sqrt.(model.layers[2].σ²)
    ps_test[2] .= model.layers[2].γ .* ps[2] ./ sqrt.(model.layers[2].σ²) - 
                    model.layers[2].γ .* model.layers[2].μ ./ sqrt.(model.layers[2].σ²) + 
                    model.layers[2].β
    ps_test[3] .= model.layers[4].γ .* ps[5] ./ sqrt.(model.layers[4].σ²)
    ps_test[4] .= model.layers[4].γ .* ps[6] ./ sqrt.(model.layers[4].σ²) - 
                    model.layers[4].γ .* model.layers[4].μ ./ sqrt.(model.layers[4].σ²) + 
                    model.layers[4].β
    ps_test[5] .= ps[9]
    ps_test[6] .= ps[10]
    nothing
end

accuracy(x, y) = begin
    ŷ = model_test(x)
    TP = sum((ŷ .≥ 0) .* (y .≥ 0))
    FP = sum((ŷ .≥ 0) .* (y .< 0))
    TN = sum((ŷ .< 0) .* (y .< 0))
    FN = sum((ŷ .< 0) .* (y .≥ 0))
    @printf("%20s%10s%10s\n", "P", "N", "Rates")
    @printf("%10s%10d%10d%10.3f\n", "P̂", TP, FP, TP/(TP+FP))
    @printf("%10s%10d%10d%10.3f\n", "N̂", FN, TN, TN/(TN+FN))
    @printf("%10s%10.3f%10.3f\n", "Rates", TP/(TP+FN), TN/(FP+TN))
    (TP+TN)/(TP+TN+FP+FN)
end

function calc_eer(x, y)::Real
    ŷ = model_test(x) |> cpu
    y = cpu(y)[:]
    taridx = findall(x -> x≥0, y)
    nonidx = findall(x -> x<0, y)
    tar = ŷ[taridx]
    non = ŷ[nonidx]
    return ROCAnalysis.eer(tar, non)
end


loss(x,y) = Flux.squared_hinge(model(x), y)
loss_test(x,y) = begin
    update_test_mod!()
    Flux.squared_hinge(model_test(x), y)
end
evalcb = () -> @show(loss_test(X_test, Y_test))

function save_epoch(μx, σx)
    update_test_mod!()
    p = ps[:] |> cpu
    BSON.@save datadir("epoch", "epoch_class.bson") p
    p = ps_test[:] |> cpu
    BSON.@save datadir("epoch", "epoch_c_test.bson") p μx σx
end

function save_params(i::Int, μx, σx, eer)
    update_test_mod!()
    p = ps_test[:] |> cpu
    BSON.@save datadir("dnn_models", "p_class$i.bson") p μx σx eer
end

function load_epoch!()
    BSON.@load datadir("epoch", "epoch_class.bson") p
    Flux.loadparams!(model, p)
    @info "Existing epoch was loaded"
end

# opt = NADAM()
opt = ADAGrad()

model = Chain(Dense(3,10), BatchNorm(10, relu), 
            Dense(10,10),BatchNorm(10, relu),
            Dense(10,1,hardtanh)) |> gpu
model_test = Chain(Dense(3,10,relu),
            Dense(10,10,relu),
            Dense(10,1,hardtanh)) |> gpu
ps = Flux.params(model)
ps_test = Flux.params(model_test)
np = sum(length, ps)

println("Do you want to load epoch? Answer 1 to load")
readline() == "1" && load_epoch!()

main()
