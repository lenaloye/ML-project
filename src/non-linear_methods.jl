### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ c348ae10-59ac-11ec-1a87-4be8f2b2490a
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, 	 
          Statistics, OpenML, MLJDecisionTreeInterface, MLJFlux, Flux
end

# ╔═╡ c78049af-c79f-4ca8-9100-456780dd18b2
md" # Non-Linear Methods

We load the precipitation data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 4118b2ba-b935-4089-8c89-e6b03e80e065
precipitation = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame);

# ╔═╡ e91f3916-6b75-4041-8b12-3b9d1fcd5e79
p = dropmissing(precipitation);

# ╔═╡ 78ca5932-9de0-4006-af40-166fc792e89c
function data_split(data;
           	             shuffle = false,
           	             idx_train = 1:1275,
           	             idx_test = 1276:1699)
      	  idxs = if shuffle
             	   randperm(size(data, 1))
         	   else
            		1:size(data, 1)
         	   end
       	 (train = data[idxs[idx_train], :],
        	test = data[idxs[idx_test], :])
end

# ╔═╡ f1be2527-d1fa-4f80-b02b-d55bffec7fd7
p1 = coerce!(p, :precipitation_nextday => Binary);

# ╔═╡ 85ff6585-0c1c-4750-a223-5c3adcfbbd1e
data1 = data_split(p1);

# ╔═╡ ae0090b3-52b9-49c8-aaca-f1de6ae59155
md" ### Tree-Based Methods"

# ╔═╡ 0b596a97-6161-4936-ba59-27124816cf63
mach = machine(RandomForestClassifier(n_trees = 500),
	         select(data1.train, Not(:precipitation_nextday)), 			
				 data1.train.precipitation_nextday) |> fit!;

# ╔═╡ 7ea486fe-d305-4a9c-b549-510231b27d20
mean(predict_mode(mach, select(data1.test, Not(:precipitation_nextday))) .== data1.test.precipitation_nextday)

# ╔═╡ 188abac4-18b7-4bc3-99af-b9cc7227ac8d
md" The test accuracy of a random forest with 500 trees is approximately 81%, better than with the linear method (74%)."

# ╔═╡ a37914a4-c299-400e-9f66-ce9415f243b1
md" ### Neural Networks"

# ╔═╡ 7db4373f-7969-4c00-9c5d-17bcc2b44a5a
mach1 = machine(NeuralNetworkClassifier(builder = MLJFlux.Short(n_hidden = 128,
                                                             dropout = 0.1,
                                                             σ = relu),
                                    batch_size = 32,
                                    epochs = 30),
             select(data1.train, Not(:precipitation_nextday)), 
					data1.train.precipitation_nextday);

# ╔═╡ a40a89ea-8275-4184-a574-9c43dcb68ca1
fit!(mach1, verbosity = 2);

# ╔═╡ a757f809-16a9-48b2-852d-2ce7572925ce
mean(predict_mode(mach1, select(data1.test, Not(:precipitation_nextday))) .== data1.test.precipitation_nextday)

# ╔═╡ Cell order:
# ╠═c348ae10-59ac-11ec-1a87-4be8f2b2490a
# ╠═c78049af-c79f-4ca8-9100-456780dd18b2
# ╠═4118b2ba-b935-4089-8c89-e6b03e80e065
# ╠═e91f3916-6b75-4041-8b12-3b9d1fcd5e79
# ╟─78ca5932-9de0-4006-af40-166fc792e89c
# ╠═f1be2527-d1fa-4f80-b02b-d55bffec7fd7
# ╠═85ff6585-0c1c-4750-a223-5c3adcfbbd1e
# ╟─ae0090b3-52b9-49c8-aaca-f1de6ae59155
# ╠═0b596a97-6161-4936-ba59-27124816cf63
# ╠═7ea486fe-d305-4a9c-b549-510231b27d20
# ╟─188abac4-18b7-4bc3-99af-b9cc7227ac8d
# ╟─a37914a4-c299-400e-9f66-ce9415f243b1
# ╠═7db4373f-7969-4c00-9c5d-17bcc2b44a5a
# ╠═a40a89ea-8275-4184-a574-9c43dcb68ca1
# ╠═a757f809-16a9-48b2-852d-2ce7572925ce
