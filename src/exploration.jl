### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ af1d09a4-3b31-4afa-a421-f733d1dc5732
begin
	using Pkg
    Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	using CSV, DataFrames, Distributions, Plots, MLJ, MLJLinearModels, Random, OpenML
end

# ╔═╡ e7567e00-b65b-4c0e-9380-1e63f0143e75
md"In this notebook we have a first look at the data set given."

# ╔═╡ daf09f40-5359-11ec-0d12-6968f789fc2b
md" # Exploration

We load the precipitation  training data from a csv file on the harddisk to a DataFrame.
Our goal is to predict whether there is some precipitation (rain, snow etc.) on the next day in Pully, getting measurements from different weather stations in Switzerland."

# ╔═╡ 29033046-8508-4eb5-9eaf-b35ac9e28569
md" ## Training data properties"

# ╔═╡ cf4b0a9b-229f-41dd-a505-271a500928f8
precipitation_training = CSV.read(joinpath(@__DIR__, "..", "data", "project", "trainingdata.csv"), DataFrame)

# ╔═╡ a06ff05b-6f70-4daa-b258-b658e5d2bff2
p_training = describe(precipitation_training)

# ╔═╡ 48337d6f-a14f-4bd2-b04f-25130027dd10
p_training[p_training.mean .== 0, :]

# ╔═╡ b290d5a7-72c2-4b3f-88a1-a0ca27e83b73
size(precipitation_training)

# ╔═╡ 003fc2b1-be15-469f-9660-ac35834a26d5
#dropmissing!(precipitation_training)  #by using this command, it will remove all the rows containing missing values. We see that it remove 1477 rows.

# ╔═╡ 1da1f4e1-70cb-420d-8f51-06439f9118c0
md" The training data contains:
- 3176 observations
- 529 predictors
The values of variables are Float64, and the value of y is a boolean. There are some missing values in the variables. We will replace them as remove them delete 1477 observations."

# ╔═╡ c862cc15-fdc6-4bbe-981a-75ce3c7b9ca7
md" ## Comparison with the other data sets"

# ╔═╡ a9cb1dde-1ad8-43ee-b3bc-a58f956ee0e3
md" #### Sample submission example"

# ╔═╡ 6f3cae44-f49b-48ca-8761-7502ea910ace
precipitation_ss = CSV.read(joinpath(@__DIR__, "..", "data", "project", "sample_submission.csv"), DataFrame)

# ╔═╡ e9f5057e-04cb-4b1a-97ae-b6f16adb8c64
size(precipitation_ss)

# ╔═╡ 6390d77d-e48e-4908-8f71-ba9dc9ab730e
md" The 2 predictors correspond to the id and the preciptation\_nextday columns. As the id does not count in the total of variables, only the precipitation\_nextday column interests us. "

# ╔═╡ 325925ad-56ad-45f2-8cd7-8b5155196c3d
md" #### Test data"

# ╔═╡ 59f7508e-f602-47e5-bdfb-1110b7fa01eb
precipitation_test = CSV.read(joinpath(@__DIR__, "..", "data", "project", "testdata.csv"), DataFrame)

# ╔═╡ 06f82640-737a-4802-ba7c-3eb5e81ee8dd
p_test = describe(precipitation_test)

# ╔═╡ 69720ba7-0f9b-47fb-b6e6-d2047dcbe6bf
p_test[p_test.mean .== 0, :]

# ╔═╡ 37fbfb84-6176-4517-b749-b1dba92deb90
size(precipitation_test)

# ╔═╡ 66aa6650-8791-4fdd-b5cf-5eb84171d508
md" #### Comparison"

# ╔═╡ 1644a087-5ab8-42d4-b1cd-ffde4384d24f
md" These values correspond to the training ones, as the sample submission data contains the precipitation_nextday variable (y), and the test data the 528 variables."

# ╔═╡ bbcb0474-f855-4291-a302-e77d9bb93ced
md" ## Precipitation_nextday properties"

# ╔═╡ d1ad87b0-ac88-40e1-8313-23c0f5a9476c
precipitation_training.precipitation_nextday[1:end]

# ╔═╡ 3e46f4ed-3493-4f47-9947-7e62c208b213
md" The y value is a boolean. Let's compare the ratio between true and false values:
#### True"

# ╔═╡ f11ecfde-026c-4c5c-9b58-1daf952fd6ea
true_val = precipitation_training[precipitation_training.precipitation_nextday .== 1, :]; # select only true

# ╔═╡ 4a548430-b87c-4aa4-93fe-dbbca7a451b7
size(true_val)

# ╔═╡ 3dcb5baf-494b-4305-9902-23b815e25550
md" #### False"

# ╔═╡ 03a42406-8d11-4b3a-b3e4-0e0f48f57c34
false_val = precipitation_training[precipitation_training.precipitation_nextday .== 0, :]; # select only false

# ╔═╡ e920e927-0ae1-40af-a199-c8cd9465a935
size(false_val)

# ╔═╡ c1738611-7b20-4f9c-bd12-6ccb98a04966
md" #### Comparison
There are 1354 true values and 1822 false ones. The total of both numbers is 3176, which corresponds to the size of training data found before. This means that there is no missing value for y."

# ╔═╡ 296df265-89c1-4936-878c-c71a2d5e8e90
begin
	p1 = bar([0, 1], [1354, 1822],
               xtick = ([0, 1], ["true", "false"]),
               xlabel = "precipitation_nextday", ylim = (0, 1900), xlim = (-.5, 1.5), label = nothing)
	plot(p1)
end

# ╔═╡ Cell order:
# ╟─e7567e00-b65b-4c0e-9380-1e63f0143e75
# ╠═af1d09a4-3b31-4afa-a421-f733d1dc5732
# ╟─daf09f40-5359-11ec-0d12-6968f789fc2b
# ╟─29033046-8508-4eb5-9eaf-b35ac9e28569
# ╠═cf4b0a9b-229f-41dd-a505-271a500928f8
# ╠═a06ff05b-6f70-4daa-b258-b658e5d2bff2
# ╠═48337d6f-a14f-4bd2-b04f-25130027dd10
# ╠═b290d5a7-72c2-4b3f-88a1-a0ca27e83b73
# ╠═003fc2b1-be15-469f-9660-ac35834a26d5
# ╟─1da1f4e1-70cb-420d-8f51-06439f9118c0
# ╟─c862cc15-fdc6-4bbe-981a-75ce3c7b9ca7
# ╟─a9cb1dde-1ad8-43ee-b3bc-a58f956ee0e3
# ╠═6f3cae44-f49b-48ca-8761-7502ea910ace
# ╠═e9f5057e-04cb-4b1a-97ae-b6f16adb8c64
# ╟─6390d77d-e48e-4908-8f71-ba9dc9ab730e
# ╟─325925ad-56ad-45f2-8cd7-8b5155196c3d
# ╠═59f7508e-f602-47e5-bdfb-1110b7fa01eb
# ╠═06f82640-737a-4802-ba7c-3eb5e81ee8dd
# ╠═69720ba7-0f9b-47fb-b6e6-d2047dcbe6bf
# ╠═37fbfb84-6176-4517-b749-b1dba92deb90
# ╟─66aa6650-8791-4fdd-b5cf-5eb84171d508
# ╟─1644a087-5ab8-42d4-b1cd-ffde4384d24f
# ╟─bbcb0474-f855-4291-a302-e77d9bb93ced
# ╠═d1ad87b0-ac88-40e1-8313-23c0f5a9476c
# ╟─3e46f4ed-3493-4f47-9947-7e62c208b213
# ╠═f11ecfde-026c-4c5c-9b58-1daf952fd6ea
# ╠═4a548430-b87c-4aa4-93fe-dbbca7a451b7
# ╟─3dcb5baf-494b-4305-9902-23b815e25550
# ╠═03a42406-8d11-4b3a-b3e4-0e0f48f57c34
# ╠═e920e927-0ae1-40af-a199-c8cd9465a935
# ╟─c1738611-7b20-4f9c-bd12-6ccb98a04966
# ╠═296df265-89c1-4936-878c-c71a2d5e8e90
