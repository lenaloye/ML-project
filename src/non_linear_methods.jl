### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 19acc8fe-636a-11ec-0930-ab87e7221b49
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, 	 
          Statistics, OpenML, MLJDecisionTreeInterface, MLJFlux, Flux, MLCourse
end

# ╔═╡ 92985e8b-5d47-4cd8-8045-85e174b70ab4
using NearestNeighborModels

# ╔═╡ c0791e81-e1fb-4f2f-918b-331c1ab86402
md" # Non-Linear Methods

We load the precipitation training and test data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 84f683ea-166a-48be-8f36-be2f673dfbbc
precipitation_training = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame);

# ╔═╡ 20264266-28d6-41e4-a741-cf257a5292ec
test_data = CSV.read(joinpath(@__DIR__, "..", "data", "project", "testdata.csv"), DataFrame);

# ╔═╡ 78e22cb7-29f6-4cbf-af15-2b5e165cd663
md"First we have to prepare our data set by filling in the missing values with some standard values with the help of `FillImputer` that fills in the median of all values. "

# ╔═╡ 1ff5f036-785c-4d92-bb50-f485653cafe6
precipitation_training_med = MLJ.transform(fit!(machine(FillImputer(), 
    			select(precipitation_training, Not(:precipitation_nextday)))), 				 
 				select(precipitation_training, Not(:precipitation_nextday)));

# ╔═╡ ca708cd5-fe0b-48ef-918f-68d6d05c0fe8
precipitation_training_med.precipitation_nextday = 
  	precipitation_training[:,:precipitation_nextday];

# ╔═╡ a7729ce7-dd4b-4cc7-9d1e-52da2644f928
training_data = coerce!(precipitation_training_med, :precipitation_nextday => Binary); # with this we tell the computer to interpret the data in column precipitation_nextday as binary data.

# ╔═╡ f57c0a0a-8c0a-44b8-b0f7-80444e66f4da
md" Then we standardize our training and test datas."

# ╔═╡ 1fde8131-6ce6-4631-baa3-d1963aa28d65
mach_train = machine(Standardizer(features=[:precipitation_nextday, :ALT_sunshine_4], ignore=true), training_data);

# ╔═╡ d67afdb4-c62e-4b6f-b9da-65fe7566be29
fit!(mach_train);

# ╔═╡ ef25fe5c-c48a-4b14-9346-a2afe272dfec
stand_train = MLJ.transform(mach_train, training_data);

# ╔═╡ 7ec88a07-05ec-4c8c-ab67-2c255167c95a
mach_test = machine(Standardizer(features=[:ZER_sunshine_1, :ABO_sunshine_4, :ALT_sunshine_4, :CHU_sunshine_4, :SAM_sunshine_4], ignore=true), test_data);

# ╔═╡ 2c3f8117-f512-4a57-90b7-c9a762c082f0
fit!(mach_test);

# ╔═╡ 48fb000c-7ec0-459b-be40-df93b0a7eee0
stand_test = MLJ.transform(mach_test, test_data);

# ╔═╡ 8b70f2c8-80bd-4287-851e-df62c49c8a4b
md"### K-Nearest-Neighbor Classification"

# ╔═╡ 8a5ab760-3b72-4998-9235-04b0c917c68c
begin
    model = KNNClassifier()
	Random.seed!(10)
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(),
                                   range = range(model, :K, values = 1:50),
                                   measure = auc)
    self_tuning_mach = machine(self_tuning_model,
                               select(stand_train, Not(:precipitation_nextday)),
                               stand_train.precipitation_nextday) |> fit!
end

# ╔═╡ 5719e72b-8eeb-41db-a7f9-5205baf9be3b
rep = report(self_tuning_mach)

# ╔═╡ 8fe64e6a-f941-48d7-bfce-58dd6584674c
scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "K", ylabel = "AUC")

# ╔═╡ 4e075958-3fa3-4c40-9910-a44dfba387bc
mach = machine(KNNClassifier(K = 22), select(stand_train, 
                Not(:precipitation_nextday)), stand_train.precipitation_nextday);

# ╔═╡ 742e3c9b-ad7b-4621-bdbe-16fedb6bab5f
fit!(mach, verbosity = 2);

# ╔═╡ 24a5da0e-8411-42fa-ad90-6a60431d73a5
md" Let's prepare these results for a submission data set. First we have to load the test set, and apply our machine on it. Then we construct our submission data and download it."

# ╔═╡ 33759d54-12ec-4b98-8bcc-8b2aaee198c4
pred = predict(mach, stand_test);

# ╔═╡ cd173cf6-7b7c-46d7-bb35-58ca031d7f6a
true_pred = pdf.(pred, true)

# ╔═╡ 43332b3e-7f32-4c8b-bbd4-b9c57b43bc04
submission = DataFrame(id = 1:1200, precipitation_nextday = true_pred);

# ╔═╡ 39c5e105-8d55-492a-b9d0-bd4356813c3a
CSV.write("../data/project/submission_knn.csv", submission);

# ╔═╡ ca26acd7-9e03-4705-a180-24e20c4895c1
md" ### Tree-Based Methods"

# ╔═╡ 571a8a39-ef01-4cf7-a2cb-35354c1e0f04
md" We first used a big interval, from 100 to 500. We found that the best model is with n_trees = 254."

# ╔═╡ fe47cb73-f752-4fbb-b857-cb98b75874d8
begin
    model1 = RandomForestClassifier()
	Random.seed!(10)
    self_tuning_model1 = TunedModel(model = model1,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(goal = 20),
                                   range = range(model1, :n_trees, scale = :log,
									       lower = 100, upper = 500),
                                   measure = auc)
    self_tuning_mach1 = machine(self_tuning_model1,
                               select(stand_train, Not(:precipitation_nextday)),
                               stand_train.precipitation_nextday) |> fit!
end

# ╔═╡ 63d19303-2195-4655-a456-f6462084e56b
rep1 = report(self_tuning_mach1)

# ╔═╡ 0505e9b6-51bd-4177-8031-c6ce2d068b05
scatter(reshape(rep1.plotting.parameter_values, :),
	    rep1.plotting.measurements, xlabel = "nb of trees", ylabel = "AUC")

# ╔═╡ e3f9c307-ce8a-4e74-804a-b01b0ed208bf
md"We decided to reduce our interval to find the best model. We obtained n_trees = 251. We continue with this value."

# ╔═╡ 0e65f610-50f6-4452-ae83-b6f3b73ef223
begin
	Random.seed!(10)
    self_tuning_model11 = TunedModel(model = model1,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(goal = 20),
                                   range = range(model1, :n_trees, scale = :log,
									       lower = 240, upper = 259),
                                   measure = auc)
    self_tuning_mach11 = machine(self_tuning_model11,
                               select(stand_train, Not(:precipitation_nextday)),
                               stand_train.precipitation_nextday) |> fit!
end

# ╔═╡ 6f6dab23-6254-4e22-b2c1-00bce00e3eaa
rep11 = report(self_tuning_mach11)

# ╔═╡ 3262d4e8-56a7-495f-84bd-8551c2f11200
scatter(reshape(rep11.plotting.parameter_values, :),
	    rep11.plotting.measurements, xlabel = "nb of trees", ylabel = "AUC")

# ╔═╡ 8758233a-2def-4149-8daf-ddb9fb61f3f7
md" Let's prepare these results for a submission data set."

# ╔═╡ 6167e652-7cfd-43a5-8cf7-e11ff6638749
mach1 = machine(RandomForestClassifier(n_trees = 251),
	         select(stand_train, Not(:precipitation_nextday)), 			
				 stand_train.precipitation_nextday);

# ╔═╡ e3c7ea50-cc6a-42ca-9500-6209451b9125
fit!(mach1, verbosity = 2);

# ╔═╡ b98737ed-f552-4d84-9032-3bf22714564d
pred1 = predict(mach1, stand_test);

# ╔═╡ ccbb089d-fa78-48f1-a430-753226936f62
true_pred1 = pdf.(pred1, true)

# ╔═╡ bd9606cf-26ba-463f-9cd0-f9c5590021c5
submission1 = DataFrame(id = 1:1200, precipitation_nextday = true_pred1);

# ╔═╡ 4fab92b5-49f7-401a-9d4f-a91fd2d03fea
CSV.write("../data/project/submission_trees.csv", submission1);

# ╔═╡ d91decea-dfd9-48a3-aeb3-3fc0a97a0a55
md" ### Neural Networks"

# ╔═╡ cecaef2e-1530-4bfc-9f48-85bafa2946d6
md" #### Two-layers"

# ╔═╡ d45c0e38-ca15-4472-addf-fded8f5d306c
mach3 = machine(NeuralNetworkClassifier(builder= MLJFlux.@builder(Chain(Dense(n_in, 50, relu), Dense(50, 50, relu),Dense(50, n_out))), batch_size = 32, epochs = 10, rng=Random.seed!(10)), select(stand_train, Not(:precipitation_nextday)), stand_train.precipitation_nextday);

# ╔═╡ 8c5c69d3-a405-42bd-a1cd-177b0548d11a
fit!(mach3, verbosity = 2);

# ╔═╡ 53fa61e9-4c69-4bdd-9abb-8a456d1ff429
md" Let's prepare these results for a submission data set."

# ╔═╡ aeff4ff4-a355-4173-81e4-0cefc3a9f0b6
pred3 = predict(mach3, stand_test);

# ╔═╡ 295c3643-daa5-4a4e-afca-85064346b100
true_pred3 = pdf.(pred3, true)

# ╔═╡ 99c0d9a4-ce42-4241-8233-1d32637aa719
submission3 = DataFrame(id = 1:1200, precipitation_nextday = true_pred3);

# ╔═╡ 957f19c0-5801-45da-9b29-0bfa29c9605f
CSV.write("../data/project/submission_nn_2_50.csv", submission3);

# ╔═╡ d3d6983b-8f7c-481e-8e8d-b754a8c64ee4
md"#### Full-connected three-layer network"

# ╔═╡ 0a34f7b8-c7d3-49c7-ad02-7ab42f2d3df3
mach2 = machine(NeuralNetworkClassifier(builder = 
 			MLJFlux.Short(n_hidden = 128, dropout = 0.1, σ = relu),
                                    batch_size = 32,
                                    epochs = 30, rng=Random.seed!(10)),
             select(stand_train, Not(:precipitation_nextday)), 			
				 stand_train.precipitation_nextday);

# ╔═╡ bc4bc2ec-cc36-493f-9111-33092f4b3b88
fit!(mach2, verbosity = 2);

# ╔═╡ 85cdf117-6f9e-46dd-b7f9-10be9a726efe
md" Let's prepare these results for a submission data set."

# ╔═╡ 15917fde-5f71-47a8-ad4c-b4a9303367fb
pred2 = predict(mach2, stand_test);

# ╔═╡ 716d82fd-efcd-4505-ba07-d9a71a483b1d
true_pred2 = pdf.(pred2, true)

# ╔═╡ 96badd40-deed-4c57-b2ca-6d99c0748e40
submission2 = DataFrame(id = 1:1200, precipitation_nextday = true_pred2);

# ╔═╡ 724aaa16-2f90-4504-8188-e20206db51d3
CSV.write("../data/project/submission_nn_128.csv", submission2);

# ╔═╡ Cell order:
# ╠═19acc8fe-636a-11ec-0930-ab87e7221b49
# ╟─c0791e81-e1fb-4f2f-918b-331c1ab86402
# ╠═84f683ea-166a-48be-8f36-be2f673dfbbc
# ╠═20264266-28d6-41e4-a741-cf257a5292ec
# ╟─78e22cb7-29f6-4cbf-af15-2b5e165cd663
# ╠═1ff5f036-785c-4d92-bb50-f485653cafe6
# ╠═ca708cd5-fe0b-48ef-918f-68d6d05c0fe8
# ╠═a7729ce7-dd4b-4cc7-9d1e-52da2644f928
# ╟─f57c0a0a-8c0a-44b8-b0f7-80444e66f4da
# ╠═1fde8131-6ce6-4631-baa3-d1963aa28d65
# ╠═d67afdb4-c62e-4b6f-b9da-65fe7566be29
# ╠═ef25fe5c-c48a-4b14-9346-a2afe272dfec
# ╠═7ec88a07-05ec-4c8c-ab67-2c255167c95a
# ╠═2c3f8117-f512-4a57-90b7-c9a762c082f0
# ╠═48fb000c-7ec0-459b-be40-df93b0a7eee0
# ╟─8b70f2c8-80bd-4287-851e-df62c49c8a4b
# ╠═92985e8b-5d47-4cd8-8045-85e174b70ab4
# ╠═8a5ab760-3b72-4998-9235-04b0c917c68c
# ╠═5719e72b-8eeb-41db-a7f9-5205baf9be3b
# ╠═8fe64e6a-f941-48d7-bfce-58dd6584674c
# ╠═4e075958-3fa3-4c40-9910-a44dfba387bc
# ╠═742e3c9b-ad7b-4621-bdbe-16fedb6bab5f
# ╟─24a5da0e-8411-42fa-ad90-6a60431d73a5
# ╠═33759d54-12ec-4b98-8bcc-8b2aaee198c4
# ╠═cd173cf6-7b7c-46d7-bb35-58ca031d7f6a
# ╠═43332b3e-7f32-4c8b-bbd4-b9c57b43bc04
# ╠═39c5e105-8d55-492a-b9d0-bd4356813c3a
# ╟─ca26acd7-9e03-4705-a180-24e20c4895c1
# ╟─571a8a39-ef01-4cf7-a2cb-35354c1e0f04
# ╠═fe47cb73-f752-4fbb-b857-cb98b75874d8
# ╠═63d19303-2195-4655-a456-f6462084e56b
# ╠═0505e9b6-51bd-4177-8031-c6ce2d068b05
# ╟─e3f9c307-ce8a-4e74-804a-b01b0ed208bf
# ╠═0e65f610-50f6-4452-ae83-b6f3b73ef223
# ╠═6f6dab23-6254-4e22-b2c1-00bce00e3eaa
# ╠═3262d4e8-56a7-495f-84bd-8551c2f11200
# ╟─8758233a-2def-4149-8daf-ddb9fb61f3f7
# ╠═6167e652-7cfd-43a5-8cf7-e11ff6638749
# ╠═e3c7ea50-cc6a-42ca-9500-6209451b9125
# ╠═b98737ed-f552-4d84-9032-3bf22714564d
# ╠═ccbb089d-fa78-48f1-a430-753226936f62
# ╠═bd9606cf-26ba-463f-9cd0-f9c5590021c5
# ╠═4fab92b5-49f7-401a-9d4f-a91fd2d03fea
# ╟─d91decea-dfd9-48a3-aeb3-3fc0a97a0a55
# ╟─cecaef2e-1530-4bfc-9f48-85bafa2946d6
# ╠═d45c0e38-ca15-4472-addf-fded8f5d306c
# ╠═8c5c69d3-a405-42bd-a1cd-177b0548d11a
# ╟─53fa61e9-4c69-4bdd-9abb-8a456d1ff429
# ╠═aeff4ff4-a355-4173-81e4-0cefc3a9f0b6
# ╠═295c3643-daa5-4a4e-afca-85064346b100
# ╠═99c0d9a4-ce42-4241-8233-1d32637aa719
# ╠═957f19c0-5801-45da-9b29-0bfa29c9605f
# ╠═d3d6983b-8f7c-481e-8e8d-b754a8c64ee4
# ╠═0a34f7b8-c7d3-49c7-ad02-7ab42f2d3df3
# ╠═bc4bc2ec-cc36-493f-9111-33092f4b3b88
# ╟─85cdf117-6f9e-46dd-b7f9-10be9a726efe
# ╠═15917fde-5f71-47a8-ad4c-b4a9303367fb
# ╠═716d82fd-efcd-4505-ba07-d9a71a483b1d
# ╠═96badd40-deed-4c57-b2ca-6d99c0748e40
# ╠═724aaa16-2f90-4504-8188-e20206db51d3
