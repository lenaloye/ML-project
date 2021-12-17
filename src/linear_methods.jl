### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 5355a88e-689c-43fc-8875-3fef921e1e98
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, 	 
          Statistics, OpenML
end

# ╔═╡ 03b21bc7-26e7-4e6d-88ec-331ba45f09d4
using NearestNeighborModels

# ╔═╡ dd4f61e2-9c47-4c04-b192-c50efbaaf22d
md" # Linear Methods

We load the precipitation data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 1174d8c0-4975-4346-a01d-32440062d9ff
precipitation = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame);

# ╔═╡ a84ff32e-0e18-4d57-adb5-6c25eaf5c635
md"First we have to prepare our data set by dropping the missing values and split the datas into a train and a test set."

# ╔═╡ 4e2e08cb-0abb-42ae-86bf-ae66fb5e2c77
p = dropmissing!(precipitation)

# ╔═╡ dd4470f7-487a-4a91-8342-7ac1e13830a5
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

# ╔═╡ d2e0e099-8736-4a19-99e0-6a0fd72db538
p1 = coerce!(p, :precipitation_nextday => Binary); # with this we tell the computer to interpret the data in column precipitation_nextday as multi-class data.

# ╔═╡ 206f4178-2ff3-40c9-902f-acce5c8e6328
data1 = data_split(p1)

# ╔═╡ ea7a929e-fccc-4407-b8b9-a3cf6653f759
md" ### Multiple Logistic Regression
Now we define a supervised learning machine. "

# ╔═╡ 9d6f9123-75af-42c2-a63b-b7db02701df3
mach = machine(LogisticClassifier(penalty = :none),
             select(data1.train, Not(:precipitation_nextday)),
             data1.train.precipitation_nextday);

# ╔═╡ 15736b3b-7929-45d7-909d-93f629454c0f
fit!(mach, verbosity = 2);

# ╔═╡ e2b88aa8-2b83-49c2-bf0e-16839cb937e0
predict(mach, select(data1.train, Not(:precipitation_nextday)))

# ╔═╡ 072ee7ed-5fc0-4284-a6f2-f7525e30af0c
confusion_matrix(predict_mode(mach, select(data1.train, Not(:precipitation_nextday))),
                 data1.train.precipitation_nextday)

# ╔═╡ ae3f269a-ed4a-4a37-a909-86367809e191
md"With our simple features, logistic regression can classify the training data  correctly. Let us see how well this works for test data.
"

# ╔═╡ 0cf9f05f-31da-4aa1-9048-450961bdd348
predict(mach, select(data1.test, Not(:precipitation_nextday)))

# ╔═╡ 8e9605ba-fcc0-4bcc-b2a4-95015aef0e31
mean(predict_mode(mach, select(data1.test, Not(:precipitation_nextday))) .== data1.test.precipitation_nextday)

# ╔═╡ 9d6e57d1-9f51-417f-9feb-f27f15c0f675
md" The test accuracy of linear classification is approximately 74%."

# ╔═╡ 011dc1f3-8331-40cb-8860-375437c3030f
confusion_matrix(predict_mode(mach, select(data1.test, Not(:precipitation_nextday))),
                 data1.test.precipitation_nextday)

# ╔═╡ da3c6be4-30d0-4cad-8c87-e70fe2985b8d
md"Let us evaluate the fit in terms of commonly used losses for binary classification."

# ╔═╡ a29f11b6-e28f-480f-811a-1d2b90762118
function losses(machine, input, response)
    (loglikelihood = -sum(log_loss(predict(machine, input), response)),
     misclassification_rate = mean(predict_mode(machine, input) .!= response),
     accuracy = accuracy(predict_mode(machine, input), response),
     auc = MLJ.auc(predict(machine, input), response)
	)
end;

# ╔═╡ 84db5e37-4459-4f1e-89d3-cc87676cf0ec
md" Let's prepare these results for a submission data set. First we have to load the test set, and apply our machine on it. Then we construct our submission data and download it."

# ╔═╡ 924f1dd5-33d4-471c-b93f-038ad24df6de
precipitation_test = CSV.read(joinpath(@__DIR__, "..", "data", "project", "testdata.csv"), DataFrame);

# ╔═╡ 28718757-38ed-4de3-b274-3d94d98a33c0
pred = predict(mach, precipitation_test)

# ╔═╡ 66a5fb2e-7f17-4779-9a9d-514801161470
true_pred = pdf.(pred, true)

# ╔═╡ 7dc95cfa-981f-4963-abe6-bceb9b7f3483
submission = DataFrame(id = 1:1200, precipitation_nextday = true_pred);

# ╔═╡ fbdd0d9f-2440-4e1c-9b40-9b115416edc9
CSV.write("../data/project/submission_regression.csv", submission)

# ╔═╡ 771b8211-139a-44c7-ad76-4672b3cb6d2e
md"### Multiple Logistic Ridge Regression"

# ╔═╡ 3ae9611c-fde4-4d63-944c-f70e4980eb4e
mach1 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2),
             select(data1.train, Not(:precipitation_nextday)),
             data1.train.precipitation_nextday);

# ╔═╡ 8932609e-d0de-41b9-92e4-b8f2e0207221
fit!(mach1, verbosity = 2);

# ╔═╡ a11b07e0-aa2e-41b1-8c40-d3bd87fe009b
predict(mach1, select(data1.train, Not(:precipitation_nextday)))

# ╔═╡ 8da552cc-35d6-477b-b22b-1d97288f4282
confusion_matrix(predict_mode(mach1, select(data1.train, Not(:precipitation_nextday))),
                 data1.train.precipitation_nextday)

# ╔═╡ 91cbbe19-f02d-4e28-ba67-b3b181d62163
predict(mach1, select(data1.test, Not(:precipitation_nextday)))

# ╔═╡ 505ab59f-65b9-402c-9a2d-d216f1534f57
mean(predict_mode(mach1, select(data1.test, Not(:precipitation_nextday))) .== data1.test.precipitation_nextday)

# ╔═╡ 0b06010d-4688-4ff4-93d6-ace0191d3b5c
md" The test accuracy of linear Ridge classification is approximately 74%."

# ╔═╡ 4d14f505-772d-4d29-ac84-8dbc6edb8e82
confusion_matrix(predict_mode(mach1, select(data1.test, Not(:precipitation_nextday))),
                 data1.test.precipitation_nextday)

# ╔═╡ d0ed0686-1680-4800-9eb7-bd5f470116f6
md" Let's prepare these results for a submission data set, same steps as the Multiple Logistic Regression."

# ╔═╡ 5567e07a-0735-4cf9-8289-81f61396cda0
pred1 = predict(mach1, precipitation_test);

# ╔═╡ 7b090cbd-6ce7-4ce8-a409-1e7d92b0ab06
true_pred1 = pdf.(pred1, true)

# ╔═╡ 72d4fd85-9d57-4d6e-a596-73bf3430d6ad
submission1 = DataFrame(id = 1:1200, precipitation_nextday = true_pred1);

# ╔═╡ 11d4fb2c-1b9e-417e-8082-9967f2414239
CSV.write("../data/project/submission_ridge_regression.csv", submission1)

# ╔═╡ e0a37c3d-7ec0-4655-9ed3-7fc5b1f32d75
md"### KNN"

# ╔═╡ aa7a4a76-7dac-4269-8fb1-ab20b9e8b563
begin
	highest_mean = 0.0
	best_k = 0
	for k in 1:100
		mach2 = machine(KNNClassifier(K = k),
             select(data1.train, Not(:precipitation_nextday)),
             data1.train.precipitation_nextday)
		fit!(mach2, verbosity = 2)
		predict(mach2, select(data1.train, Not(:precipitation_nextday)))
		mean2 = mean(predict_mode(mach2, select(data1.test, 
                Not(:precipitation_nextday))) .== data1.test.precipitation_nextday)
		if (mean2 > highest_mean)
			highest_mean = mean2
			best_k = k
		end
	end
end

# ╔═╡ 278bdb5b-ba42-4f58-bebc-a09d7f8b77e0
highest_mean, best_k

# ╔═╡ 7ac13cb6-7e6b-4371-82a0-6790b060d1d4
mach2 = machine(KNNClassifier(K = 12),
             select(data1.train, Not(:precipitation_nextday)),
             data1.train.precipitation_nextday);

# ╔═╡ 3798f993-c986-4d27-be62-a51be5aec6e6
fit!(mach2, verbosity = 2);

# ╔═╡ 3d9bbf32-246f-44ac-8b52-691850acf22b
pred2 = predict(mach2, precipitation_test);

# ╔═╡ 82d0e431-c488-415b-8b82-36535c232bd7
true_pred2 = pdf.(pred2, true)

# ╔═╡ 47a9be3a-3164-4d95-9f8f-8e2f403f919d
submission2 = DataFrame(id = 1:1200, precipitation_nextday = true_pred2);

# ╔═╡ eba3b687-f059-4048-ad84-975783e29afe
CSV.write("../data/project/submission_knn.csv", submission2)

# ╔═╡ Cell order:
# ╠═5355a88e-689c-43fc-8875-3fef921e1e98
# ╟─dd4f61e2-9c47-4c04-b192-c50efbaaf22d
# ╠═1174d8c0-4975-4346-a01d-32440062d9ff
# ╟─a84ff32e-0e18-4d57-adb5-6c25eaf5c635
# ╠═4e2e08cb-0abb-42ae-86bf-ae66fb5e2c77
# ╠═dd4470f7-487a-4a91-8342-7ac1e13830a5
# ╠═d2e0e099-8736-4a19-99e0-6a0fd72db538
# ╠═206f4178-2ff3-40c9-902f-acce5c8e6328
# ╟─ea7a929e-fccc-4407-b8b9-a3cf6653f759
# ╠═9d6f9123-75af-42c2-a63b-b7db02701df3
# ╠═15736b3b-7929-45d7-909d-93f629454c0f
# ╠═e2b88aa8-2b83-49c2-bf0e-16839cb937e0
# ╠═072ee7ed-5fc0-4284-a6f2-f7525e30af0c
# ╟─ae3f269a-ed4a-4a37-a909-86367809e191
# ╠═0cf9f05f-31da-4aa1-9048-450961bdd348
# ╠═8e9605ba-fcc0-4bcc-b2a4-95015aef0e31
# ╟─9d6e57d1-9f51-417f-9feb-f27f15c0f675
# ╠═011dc1f3-8331-40cb-8860-375437c3030f
# ╟─da3c6be4-30d0-4cad-8c87-e70fe2985b8d
# ╠═a29f11b6-e28f-480f-811a-1d2b90762118
# ╠═84db5e37-4459-4f1e-89d3-cc87676cf0ec
# ╠═924f1dd5-33d4-471c-b93f-038ad24df6de
# ╠═28718757-38ed-4de3-b274-3d94d98a33c0
# ╠═66a5fb2e-7f17-4779-9a9d-514801161470
# ╠═7dc95cfa-981f-4963-abe6-bceb9b7f3483
# ╠═fbdd0d9f-2440-4e1c-9b40-9b115416edc9
# ╟─771b8211-139a-44c7-ad76-4672b3cb6d2e
# ╠═3ae9611c-fde4-4d63-944c-f70e4980eb4e
# ╠═8932609e-d0de-41b9-92e4-b8f2e0207221
# ╠═a11b07e0-aa2e-41b1-8c40-d3bd87fe009b
# ╠═8da552cc-35d6-477b-b22b-1d97288f4282
# ╠═91cbbe19-f02d-4e28-ba67-b3b181d62163
# ╠═505ab59f-65b9-402c-9a2d-d216f1534f57
# ╟─0b06010d-4688-4ff4-93d6-ace0191d3b5c
# ╠═4d14f505-772d-4d29-ac84-8dbc6edb8e82
# ╟─d0ed0686-1680-4800-9eb7-bd5f470116f6
# ╠═5567e07a-0735-4cf9-8289-81f61396cda0
# ╠═7b090cbd-6ce7-4ce8-a409-1e7d92b0ab06
# ╠═72d4fd85-9d57-4d6e-a596-73bf3430d6ad
# ╠═11d4fb2c-1b9e-417e-8082-9967f2414239
# ╟─e0a37c3d-7ec0-4655-9ed3-7fc5b1f32d75
# ╠═03b21bc7-26e7-4e6d-88ec-331ba45f09d4
# ╠═aa7a4a76-7dac-4269-8fb1-ab20b9e8b563
# ╠═278bdb5b-ba42-4f58-bebc-a09d7f8b77e0
# ╠═7ac13cb6-7e6b-4371-82a0-6790b060d1d4
# ╠═3798f993-c986-4d27-be62-a51be5aec6e6
# ╠═3d9bbf32-246f-44ac-8b52-691850acf22b
# ╠═82d0e431-c488-415b-8b82-36535c232bd7
# ╠═47a9be3a-3164-4d95-9f8f-8e2f403f919d
# ╠═eba3b687-f059-4048-ad84-975783e29afe
