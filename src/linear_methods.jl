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

# ╔═╡ dd4f61e2-9c47-4c04-b192-c50efbaaf22d
md" # Linear Methods

We load the precipitation data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 1174d8c0-4975-4346-a01d-32440062d9ff
precipitation = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame);

# ╔═╡ a84ff32e-0e18-4d57-adb5-6c25eaf5c635
md"First we have to prepare our data set by dropping the missing values and split the datas into a train and a test set."

# ╔═╡ 4e2e08cb-0abb-42ae-86bf-ae66fb5e2c77
p = dropmissing(precipitation)

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
             data1.train.precipitation_nextday) |> fit!;

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

# ╔═╡ 27d3089f-edf9-4757-89d9-03d4138d7ff8
losses(mach, select(data1.train, Not(:precipitation_nextday)),
                 data1.train.precipitation_nextday)

# ╔═╡ 2c433397-473c-4579-af34-a64fd9298695
losses(mach, select(data1.test, Not(:precipitation_nextday)),
                 data1.test.precipitation_nextday)

# ╔═╡ 771b8211-139a-44c7-ad76-4672b3cb6d2e
md"### Multiple Logistic Ridge Regression"

# ╔═╡ 3ae9611c-fde4-4d63-944c-f70e4980eb4e
mach1 = machine(LogisticClassifier(penalty = :l2, lambda = 2e-2),
             select(data1.train, Not(:precipitation_nextday)),
             data1.train.precipitation_nextday) |> fit!;

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

# ╔═╡ f81fa703-8b51-4fa5-935f-c4a6db0763e9
md"We see that the test misclassification rate with regularization
is lower than in our original fit without regularization. The misclassification rate on the training set is higher. This indicates that unregularized logistic regression is too flexible for our data set.
"

# ╔═╡ 32e39896-f6cf-49a6-9217-b1845b533839
losses(mach1, select(data1.train, Not(:precipitation_nextday)),
                 data1.train.precipitation_nextday)

# ╔═╡ 717913f7-18aa-4f45-bf03-f3077f037d75
losses(mach1, select(data1.test, Not(:precipitation_nextday)),
                 data1.test.precipitation_nextday)

# ╔═╡ Cell order:
# ╠═5355a88e-689c-43fc-8875-3fef921e1e98
# ╠═dd4f61e2-9c47-4c04-b192-c50efbaaf22d
# ╠═1174d8c0-4975-4346-a01d-32440062d9ff
# ╟─a84ff32e-0e18-4d57-adb5-6c25eaf5c635
# ╠═4e2e08cb-0abb-42ae-86bf-ae66fb5e2c77
# ╠═dd4470f7-487a-4a91-8342-7ac1e13830a5
# ╠═d2e0e099-8736-4a19-99e0-6a0fd72db538
# ╠═206f4178-2ff3-40c9-902f-acce5c8e6328
# ╟─ea7a929e-fccc-4407-b8b9-a3cf6653f759
# ╠═9d6f9123-75af-42c2-a63b-b7db02701df3
# ╠═e2b88aa8-2b83-49c2-bf0e-16839cb937e0
# ╠═072ee7ed-5fc0-4284-a6f2-f7525e30af0c
# ╟─ae3f269a-ed4a-4a37-a909-86367809e191
# ╠═0cf9f05f-31da-4aa1-9048-450961bdd348
# ╠═8e9605ba-fcc0-4bcc-b2a4-95015aef0e31
# ╟─9d6e57d1-9f51-417f-9feb-f27f15c0f675
# ╠═011dc1f3-8331-40cb-8860-375437c3030f
# ╟─da3c6be4-30d0-4cad-8c87-e70fe2985b8d
# ╠═a29f11b6-e28f-480f-811a-1d2b90762118
# ╠═27d3089f-edf9-4757-89d9-03d4138d7ff8
# ╠═2c433397-473c-4579-af34-a64fd9298695
# ╟─771b8211-139a-44c7-ad76-4672b3cb6d2e
# ╠═3ae9611c-fde4-4d63-944c-f70e4980eb4e
# ╠═a11b07e0-aa2e-41b1-8c40-d3bd87fe009b
# ╠═8da552cc-35d6-477b-b22b-1d97288f4282
# ╠═91cbbe19-f02d-4e28-ba67-b3b181d62163
# ╠═505ab59f-65b9-402c-9a2d-d216f1534f57
# ╟─0b06010d-4688-4ff4-93d6-ace0191d3b5c
# ╠═4d14f505-772d-4d29-ac84-8dbc6edb8e82
# ╟─f81fa703-8b51-4fa5-935f-c4a6db0763e9
# ╠═32e39896-f6cf-49a6-9217-b1845b533839
# ╠═717913f7-18aa-4f45-bf03-f3077f037d75
