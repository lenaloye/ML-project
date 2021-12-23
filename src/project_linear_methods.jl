### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 44f7065e-6361-11ec-1eb9-87eede3d3e19
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, 	 
          Statistics, OpenML
end

# ╔═╡ 0d92a992-f38b-4b3c-b6a1-26f8a84caa0a
md" # Linear Methods

We load the precipitation training and test data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 21ede700-3339-4239-b621-59b2b16c623a
precipitation_training = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame);

# ╔═╡ c272f937-d1cc-4ed7-a0f7-761ca6dd851c
test_data = CSV.read(joinpath(@__DIR__, "..", "data", "project", "testdata.csv"), DataFrame);

# ╔═╡ 2ff5563d-64cb-4a6e-b145-deb599cc92fe
md"First we have to prepare our data set by filling in the missing values with some standard values with the help of `FillImputer` that fills in the median of all values. "

# ╔═╡ d15a02bb-8a63-411a-83e8-a9f1f12d2112
precipitation_training_med = MLJ.transform(fit!(machine(FillImputer(), 
    			select(precipitation_training, Not(:precipitation_nextday)))), 				 
 				select(precipitation_training, Not(:precipitation_nextday)));

# ╔═╡ cf343f0c-fd0a-4826-aeb6-c09df22849f9
precipitation_training_med.precipitation_nextday = 
  	precipitation_training[:,:precipitation_nextday];

# ╔═╡ f41654cb-5b1d-4fbf-96cf-27a61a4d128f
training_data = coerce!(precipitation_training_med, :precipitation_nextday => Binary); # with this we tell the computer to interpret the data in column precipitation_nextday as binary data.

# ╔═╡ c376c70e-09cd-4ad2-9cbb-597c05cc4060
md" Then we standardize our training and test data sets."

# ╔═╡ 5c0bd65f-757f-492e-bb8f-afac8f78b077
mach_train = machine(Standardizer(features=[:precipitation_nextday, :ALT_sunshine_4], ignore=true), training_data);

# ╔═╡ 3bd0aa5f-e8fa-4dbf-8148-b2558a3fd702
fit!(mach_train);

# ╔═╡ e814afee-7c16-47ff-9cdd-e907b1d98d74
stand_train = MLJ.transform(mach_train, training_data)

# ╔═╡ 313e00ac-f56e-4f25-b52a-048b4723e6bb
mach_test = machine(Standardizer(features=[:ZER_sunshine_1, :ABO_sunshine_4, :ALT_sunshine_4, :CHU_sunshine_4, :SAM_sunshine_4], ignore=true), test_data);

# ╔═╡ d4833fce-3ba1-424f-82bb-3bc577f69a4b
fit!(mach_test);

# ╔═╡ d5819218-3452-49de-be58-a50b034dc1cd
stand_test = MLJ.transform(mach_test, test_data)

# ╔═╡ 91ad48cd-291f-425b-b0ec-661e18b9ca1e
md" ## Logistic Regression 
### Lasso regularization
Now we define a supervised learning machine and tune the hyper-parameters.
We first try with an interval from 1e-2 to 10 and find a lambda of approximately 4.3."

# ╔═╡ 4952b817-ddb8-48e6-b849-283b46826815
begin
    model = LogisticClassifier(penalty = :l1)
	Random.seed!(10)
    self_tuning_model0 = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(goal = 50),
                                   range = range(model, :lambda,
									       scale = :log,
									       lower = 1e-2, upper = 10),
									       measure = auc)
    self_tuning_mach0 = machine(self_tuning_model0, select(stand_train, 		 
              Not(:precipitation_nextday)), 		 
              stand_train.precipitation_nextday)  |> fit!
end

# ╔═╡ 0417d37a-5b5f-4f5f-973c-14dfa0750245
rep0 = report(self_tuning_mach0)

# ╔═╡ 0a4a9e81-f557-4e9a-a4ca-4c317a3ad715
md"We decided to reduce our interval to find the best lambda."

# ╔═╡ 32647bb8-e59b-4ed7-bcbc-133af5d510c7
begin
	Random.seed!(10)
    self_tuning_model = TunedModel(model = model,
                                   resampling = CV(nfolds = 5),
                                   tuning = Grid(goal = 50),
                                   range = range(model, :lambda,
									       scale = :log,
									       lower = 3, upper = 4.5),
									       measure = auc)
    self_tuning_mach = machine(self_tuning_model, select(stand_train, 		 
              Not(:precipitation_nextday)), 		 
              stand_train.precipitation_nextday)  |> fit!
end

# ╔═╡ 6417b449-cf76-410d-ba15-3ccca286d7dc
rep = report(self_tuning_mach)

# ╔═╡ 7a4d88c2-2d09-4c9c-ae66-4c5a61a7a858
scatter(reshape(rep.plotting.parameter_values, :),
	    rep.plotting.measurements, xlabel = "lamda", ylabel = "AUC")

# ╔═╡ 84bda3e0-f8db-4985-b25d-e05c63772b35
mach = machine(LogisticClassifier(penalty = :l1, lambda = 4.00775),
             select(stand_train, Not(:precipitation_nextday)),
             stand_train.precipitation_nextday);

# ╔═╡ 2172f78d-d189-419e-839c-ef4844f94969
fit!(mach, verbosity = 2);

# ╔═╡ b7b22d76-2911-4507-9095-5c936da77bbd
md" Let's prepare these results for a submission data set. First we have to apply our machine on the test data. Then we construct our submission data and download it."

# ╔═╡ dee42776-170d-46cb-a194-946cc66ef9b7
pred = predict(mach, stand_test);

# ╔═╡ b847717d-310b-4d79-835f-a3ab2ee5cfed
true_pred = pdf.(pred, true)

# ╔═╡ fe1cf000-7f98-4d70-954a-5e436fb86581
submission = DataFrame(id = 1:1200, precipitation_nextday = true_pred);

# ╔═╡ f3a01c5e-4d97-44fb-94a7-59e4639edbcc
CSV.write("../data/project/submission_regression.csv", submission);

# ╔═╡ Cell order:
# ╠═44f7065e-6361-11ec-1eb9-87eede3d3e19
# ╟─0d92a992-f38b-4b3c-b6a1-26f8a84caa0a
# ╠═21ede700-3339-4239-b621-59b2b16c623a
# ╠═c272f937-d1cc-4ed7-a0f7-761ca6dd851c
# ╟─2ff5563d-64cb-4a6e-b145-deb599cc92fe
# ╠═d15a02bb-8a63-411a-83e8-a9f1f12d2112
# ╠═cf343f0c-fd0a-4826-aeb6-c09df22849f9
# ╠═f41654cb-5b1d-4fbf-96cf-27a61a4d128f
# ╟─c376c70e-09cd-4ad2-9cbb-597c05cc4060
# ╠═5c0bd65f-757f-492e-bb8f-afac8f78b077
# ╠═3bd0aa5f-e8fa-4dbf-8148-b2558a3fd702
# ╠═e814afee-7c16-47ff-9cdd-e907b1d98d74
# ╠═313e00ac-f56e-4f25-b52a-048b4723e6bb
# ╠═d4833fce-3ba1-424f-82bb-3bc577f69a4b
# ╠═d5819218-3452-49de-be58-a50b034dc1cd
# ╟─91ad48cd-291f-425b-b0ec-661e18b9ca1e
# ╠═4952b817-ddb8-48e6-b849-283b46826815
# ╠═0417d37a-5b5f-4f5f-973c-14dfa0750245
# ╟─0a4a9e81-f557-4e9a-a4ca-4c317a3ad715
# ╠═32647bb8-e59b-4ed7-bcbc-133af5d510c7
# ╠═6417b449-cf76-410d-ba15-3ccca286d7dc
# ╟─7a4d88c2-2d09-4c9c-ae66-4c5a61a7a858
# ╠═84bda3e0-f8db-4985-b25d-e05c63772b35
# ╠═2172f78d-d189-419e-839c-ef4844f94969
# ╟─b7b22d76-2911-4507-9095-5c936da77bbd
# ╠═dee42776-170d-46cb-a194-946cc66ef9b7
# ╠═b847717d-310b-4d79-835f-a3ab2ee5cfed
# ╠═fe1cf000-7f98-4d70-954a-5e436fb86581
# ╠═f3a01c5e-4d97-44fb-94a7-59e4639edbcc
