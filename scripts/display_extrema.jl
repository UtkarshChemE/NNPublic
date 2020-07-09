using DrWatson
@quickactivate "NNFitter"

using Printf
using Statistics: mean, std
import ROCAnalysis
import BSON

hardtanh(x) = max(-1, min(1, x))

function main()
    segNum = parse(Int, ARGS[1])

    BSON.@load datadir("xyvals", "xyvals$segNum.bson") xydata
    @info "Loaded Learning data"
    xydata = permutedims(xydata)
    if isfile(datadir("dnn_models", "p_class$segNum.bson"))
        BSON.@load datadir("dnn_models", "p_class$segNum.bson") p μx σx
        # Creating normalized(X) and Y
        X = xydata[1:3, :]
        taridx = findall(x->0.5<x<1.5, xydata[4, :])
        nonidx = findall(x->x>1.5, xydata[4, :])
        X = (X .- μx) ./ σx 

        X1 = p[1]*X .+ p[2]
        X1a = max.(0, X1)
        X2 = p[3]*X1a .+ p[4]
        X2a = max.(0, X2)
        Y = p[5]*X2a .+ p[6]
        tar = Y[taridx]
        non = Y[nonidx]
        EER = ROCAnalysis.eer(tar, non)
        EERch = ROCAnalysis.eerch(tar, non)
        myEER = ROCAnalysis.eer(hardtanh.(tar), hardtanh.(non))
        myEERch = ROCAnalysis.eerch(hardtanh.(tar), hardtanh.(non))
        println("Extrema of X1 for classification")
        show(stdout, "text/plain", myextrema(X1; dims=2))
        println("\nExtrema of X2 for classification")
        show(stdout, "text/plain", myextrema(X2; dims=2))
        println("\nExtrema of Y for classification")
        show(stdout, "text/plain", myextrema(Y; dims=2))
        print("\nThe crossover score is $EER and $EERch")
        print("\nThe crossover score for tanh is $myEER and $myEERch")
    end

    BSON.@load datadir("dnn_models", "p_reg$segNum.bson") p μx σx

    # Creating normalized(X) and Y
    X = xydata[1:3, :]
    X = (X .- μx) ./ σx 

    X1 = p[1]*X .+ p[2]
    X1a = max.(0, X1)
    X2 = p[3]*X1a .+ p[4]
    X2a = max.(0, X2)
    Y = p[5]*X2a .+ p[6]
    println("\nExtrema of X1 for regression")
    show(stdout, "text/plain", myextrema(X1; dims=2))
    println("\nExtrema of X2 for regression")
    show(stdout, "text/plain", myextrema(X2; dims=2))
    println("\nExtrema of Y for regression")
    show(stdout, "text/plain", myextrema(Y; dims=2))
end #main

function myextrema(x; dims=2)
    b = extrema(x; dims=dims)
    return permutedims(reduce(hcat, reduce.(vcat, b)))
end
 
main()
