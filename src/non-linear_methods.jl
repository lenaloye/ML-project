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

# ╔═╡ 6df4325d-1e9a-4d53-9b0c-510ab2ca4fcb
using NearestNeighborModels

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

# ╔═╡ 85acacdb-0f1d-4a4d-8c52-a22aeb950fbe
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

# ╔═╡ d898eb8c-e443-458d-93d5-aef33d9b654b
md"### K-Nearest-Neighbor Classification"

# ╔═╡ 2e320999-ceff-41b2-821b-4f73faded635
begin
    model = KNNClassifier()
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :K, values = 1:50),
                                   measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                               select(data1.train, Not(:precipitation_nextday)),
                               data1.train.precipitation_nextday) |> fit!
end

# ╔═╡ 2f6c0f49-0ab8-4b29-a82a-7794c56f311d
rep = report(self_tuning_mach)

# ╔═╡ 000c23d9-7a89-4872-804f-e761b2d00039
scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "K", ylabel = "AUC")

# ╔═╡ 4a6c9c0d-318a-4eb7-8c78-222fbdaff9cf
mach2 = machine(KNNClassifier(K = 28), select(data1.train, 
                Not(:precipitation_nextday)), data1.train.precipitation_nextday);

# ╔═╡ 0b6ab4dd-6f2e-4d44-aa1c-5361cb0f8150
fit!(mach2, verbosity = 2);

# ╔═╡ 6780de65-22a8-4d15-97f2-e0ab0a715c93
losses(mach2, select(data1.test, Not(:precipitation_nextday)),  		 
        data1.test.precipitation_nextday)

# ╔═╡ bc569bde-a12c-4fea-a546-f185f6bf8591
md" The test accuracy of KNN is approximately 80%, and the AUC is 88.7%."

# ╔═╡ ee72c43b-580c-4808-b4f0-e3f5648ced04
md" Let's prepare these results for a submission data set. First we have to load the test set, and apply our machine on it. Then we construct our submission data and download it."

# ╔═╡ 0b2776b8-2d91-49a8-a84b-f9133f23c289
precipitation_test = CSV.read(joinpath(@__DIR__, "..", "data", "project", "testdata.csv"), DataFrame);

# ╔═╡ 2e42d8ed-65c5-4204-9dd4-ed66be3eec5c
pred2 = predict(mach2, precipitation_test);

# ╔═╡ 047669ea-b704-43d9-be00-854f9eed20ae
true_pred2 = pdf.(pred2, true)

# ╔═╡ 81199ec1-a198-46a0-adce-76bbc96e44c7
submission2 = DataFrame(id = 1:1200, precipitation_nextday = true_pred2);

# ╔═╡ d739f23b-b582-44ca-844a-f2c1974d0b90
CSV.write("../data/project/submission_knn2.csv", submission2);

# ╔═╡ ae0090b3-52b9-49c8-aaca-f1de6ae59155
md" ### Tree-Based Methods"

# ╔═╡ 0b596a97-6161-4936-ba59-27124816cf63
mach = machine(RandomForestClassifier(n_trees = 500),
	         select(data1.train, Not(:precipitation_nextday)), 			
				 data1.train.precipitation_nextday);

# ╔═╡ 351e62a0-cc1a-4de1-9956-e496d1977b9c
fit!(mach, verbosity = 2);

# ╔═╡ 7ea486fe-d305-4a9c-b549-510231b27d20
losses(mach, select(data1.test, Not(:precipitation_nextday)),  		 
        data1.test.precipitation_nextday)

# ╔═╡ 188abac4-18b7-4bc3-99af-b9cc7227ac8d
md" The test accuracy of a random forest with 500 trees is approximately 81%, and the AUC is 91%."

# ╔═╡ bca7d6dd-9e78-4315-b79b-ff5dacf1d2cb
md" Let's prepare these results for a submission data set."

# ╔═╡ f84b6252-f1bc-4458-9d11-26dce5390f65
pred = predict(mach, precipitation_test);

# ╔═╡ d5738b31-e26e-48a2-a911-0577601aa7f6
true_pred = pdf.(pred, true)

# ╔═╡ 10e89d0c-d8e7-4dbf-93cd-0ecc98cb8073
submission = DataFrame(id = 1:1200, precipitation_nextday = true_pred);

# ╔═╡ 5509caa0-531d-41a3-b5bd-e09b369fbac5
CSV.write("../data/project/submission_tree.csv", submission);

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
losses(mach1, select(data1.test, Not(:precipitation_nextday)),  		 
        data1.test.precipitation_nextday)

# ╔═╡ 36105cfe-8679-44aa-85e1-73f4ce431b55
md" The test accuracy of a neural network is approximately 77%, and the AUC is 80.7%."

# ╔═╡ 70e65fbd-972c-4e70-bd15-0e2a40a74d49
pred1 = predict(mach1, precipitation_test)

# ╔═╡ 767e945e-47ae-43a0-bc8a-de3630daf66b
true_pred1 = pdf.(pred1, true)

# ╔═╡ b5ed81b0-8f0f-45d8-a493-db31a3cc4c74
submission1 = DataFrame(id = 1:1200, precipitation_nextday = true_pred1);

# ╔═╡ 300d4d2b-d8a4-4b8c-a116-49fb933251c2
CSV.write("../data/project/submission_neural.csv", submission1);

# ╔═╡ Cell order:
# ╠═c348ae10-59ac-11ec-1a87-4be8f2b2490a
# ╟─c78049af-c79f-4ca8-9100-456780dd18b2
# ╠═4118b2ba-b935-4089-8c89-e6b03e80e065
# ╠═e91f3916-6b75-4041-8b12-3b9d1fcd5e79
# ╟─78ca5932-9de0-4006-af40-166fc792e89c
# ╠═f1be2527-d1fa-4f80-b02b-d55bffec7fd7
# ╠═85ff6585-0c1c-4750-a223-5c3adcfbbd1e
# ╠═85acacdb-0f1d-4a4d-8c52-a22aeb950fbe
# ╟─d898eb8c-e443-458d-93d5-aef33d9b654b
# ╠═6df4325d-1e9a-4d53-9b0c-510ab2ca4fcb
# ╠═2e320999-ceff-41b2-821b-4f73faded635
# ╠═2f6c0f49-0ab8-4b29-a82a-7794c56f311d
# ╠═000c23d9-7a89-4872-804f-e761b2d00039
# ╠═4a6c9c0d-318a-4eb7-8c78-222fbdaff9cf
# ╠═0b6ab4dd-6f2e-4d44-aa1c-5361cb0f8150
# ╠═6780de65-22a8-4d15-97f2-e0ab0a715c93
# ╟─bc569bde-a12c-4fea-a546-f185f6bf8591
# ╟─ee72c43b-580c-4808-b4f0-e3f5648ced04
# ╠═0b2776b8-2d91-49a8-a84b-f9133f23c289
# ╠═2e42d8ed-65c5-4204-9dd4-ed66be3eec5c
# ╠═047669ea-b704-43d9-be00-854f9eed20ae
# ╠═81199ec1-a198-46a0-adce-76bbc96e44c7
# ╠═d739f23b-b582-44ca-844a-f2c1974d0b90
# ╟─ae0090b3-52b9-49c8-aaca-f1de6ae59155
# ╠═0b596a97-6161-4936-ba59-27124816cf63
# ╠═351e62a0-cc1a-4de1-9956-e496d1977b9c
# ╠═7ea486fe-d305-4a9c-b549-510231b27d20
# ╟─188abac4-18b7-4bc3-99af-b9cc7227ac8d
# ╟─bca7d6dd-9e78-4315-b79b-ff5dacf1d2cb
# ╠═f84b6252-f1bc-4458-9d11-26dce5390f65
# ╠═d5738b31-e26e-48a2-a911-0577601aa7f6
# ╠═10e89d0c-d8e7-4dbf-93cd-0ecc98cb8073
# ╠═5509caa0-531d-41a3-b5bd-e09b369fbac5
# ╟─a37914a4-c299-400e-9f66-ce9415f243b1
# ╠═7db4373f-7969-4c00-9c5d-17bcc2b44a5a
# ╠═a40a89ea-8275-4184-a574-9c43dcb68ca1
# ╠═a757f809-16a9-48b2-852d-2ce7572925ce
# ╟─36105cfe-8679-44aa-85e1-73f4ce431b55
# ╠═70e65fbd-972c-4e70-bd15-0e2a40a74d49
# ╠═767e945e-47ae-43a0-bc8a-de3630daf66b
# ╠═b5ed81b0-8f0f-45d8-a493-db31a3cc4c74
# ╠═300d4d2b-d8a4-4b8c-a116-49fb933251c2
