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
using Plots, StatsPlots
import BSON
pyplot()


function main()
    segNum = parse(Int, ARGS[1])
    epochs = parse(Int, ARGS[2])

    BSON.@load datadir("xyvals", "xyvals$segNum.bson") xydata
    @info "Loaded Learning data"
    xydata = permutedims(xydata)

    # Creating normalized(X) and Y
    idx = shuffle(findall(x-> 0.5 < x < 1.5, xydata[4,:])) #Eliminate all none solved or infeasible
    println("Acceptable points are ", length(idx))
    X = xydata[1:3, idx]
    Y = xydata[5:5, idx]
    μx = mean(X; dims=2)
    σx = std(X; dims=2, mean=μx, corrected=false)
    X = (X .- μx) ./ σx .* (σx .> 1e-4) |> gpu
    μy = mean(Y; dims=2)
    σy = std(Y; dims=2, mean=μy, corrected=false)
    Y = (Y .- μy) ./ σy .* (σy .> 1e-4) |> gpu
    
    # Chunking and creating a balanced dataset
    # train_idx = idx[1001:end]
    # test_idx = idx[1:1000]
    
    #Creating Training and testing data
    X_train = X[1:3, 1001:end] |> gpu
    Y_train = Y[1:1, 1001:end] |> gpu
    X_test = X[1:3, 1:1000] |> gpu
    Y_test = Y[1:1, 1:1000] |> gpu

    train_loader = DataLoader(X_train, Y_train; batchsize = 2^13)
    println("loss(xtrain, ytrain) = ", loss_test(X_train, Y_train), 
            " loss(xtest, ytest) = ", loss_test(X_test,Y_test))
    @info "Training data created. Training Starts"
    plot_epoch(X_test, Y_test, 0)
    Flux.train!(loss, ps, train_loader, opt)
    
    for i=1:epochs
        @info "Starting epoch $i"
        Flux.train!(loss, ps, ncycle(train_loader, 25), opt)
        if i%20==0
            println("Epoch Number is $i")
            println("loss(xtrain, ytrain) = ", loss_test(X_train, Y_train), 
                " loss(xtest, ytest) = ", loss_test(X_test,Y_test))
            save_epoch(μx, σx, μy, σy)
            plot_epoch(X_test, Y_test, i)
            println("Should I stop? answer 1 to stop")
            if readline() == "1"
                break
            end
        end
    end
    save_params(segNum, μx, σx, μy, σy)
    plot_ends(X, Y, segNum)
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

loss(x,y) = Flux.mse(model(x), y)
loss_test(x,y) = begin
    update_test_mod!()
    Flux.mse(model_test(x), y)
end
evalcb = () -> @show(loss_test(X_test, Y_test))

function plot_epoch(x, y, i)::Nothing
    P = scatter(cpu(y)[:], cpu(model_test(x))[:], α = 0.5, legend = false)
    plot!(P, cpu(y)[:], cpu(y)[:])
    savefig(P, plotsdir("epoch", "epoch$i.png"))
    plotpath = plotsdir("epoch", "epoch$i.png")
    run(`viu $plotpath`)
    nothing
end

function plot_ends(x, y, i)::Nothing
    P = scatter(cpu(y)[:], cpu(model_test(x))[:], α = 0.5, legend = false)
    plot!(P, cpu(y)[:], cpu(y)[:])
    savefig(P, plotsdir("end$i.png"))
    plotpath = plotsdir("end$i.png")
    run(`viu $plotpath`)
    nothing
end

function save_epoch(μx, σx, μy, σy)
    update_test_mod!()
    p = ps[:] |> cpu
    BSON.@save datadir("epoch", "epoch_reg.bson") p
    p = ps_test[:] |> cpu
    BSON.@save datadir("epoch", "epoch_r_test.bson") p μx σx μy σy
end

function save_params(i::Int, μx, σx, μy, σy)
    update_test_mod!()
    p = ps_test[:] |> cpu
    BSON.@save datadir("dnn_models", "p_reg$i.bson") p μx σx μy σy
end

function load_epoch!()
    BSON.@load datadir("epoch", "epoch_reg.bson") p
    Flux.loadparams!(model, p)
    @info "Existing epoch was loaded"
end
 
opt = NADAM()
# opt = ADAGrad()

model = Chain(Dense(3,10), BatchNorm(10, relu), 
            Dense(10,10),BatchNorm(10, relu),
            Dense(10,1)) |> gpu
model_test = Chain(Dense(3,10,relu),
            Dense(10,10,relu),
            Dense(10,1)) |> gpu
ps = Flux.params(model)
ps_test = Flux.params(model_test)
np = sum(length, ps)

println("Do you want to load epoch? Answer 1 to load")
readline() == "1" && load_epoch!()

main()
