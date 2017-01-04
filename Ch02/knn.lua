local matrix = require "matrix"
local var2str = require "var2str"

local knn = {}
local mt = {}

function knn.create()
	local ml = {}
	ml.training_label = {}
	ml.training_data = {}

	setmetatable(ml, {__index= mt})
	return ml
end

function mt:add_training(label, vector)
	assert(type(vector) == "table")

	table.insert(self.training_label, label)
	table.insert(self.training_data, vector)
end

function mt:save(path)
	local file = assert(io.open(path, "w"))
	file:write(string.format("local training_label = %s;\n", var2str(self.training_label)))
	file:write(string.format("local training_data = %s;\n", var2str(self.training_data)))
	file:write("return training_label, training_data")
	file:close()
end

function mt:load(path)
	local file = io.open(path, "r")
	if not file then
		return false
	end
	local content = file:read("a")
	local fn = load(content)
	local training_label, training_data = fn()
	assert(#training_label == #training_data)
	self.training_label = training_label
	self.training_data = training_data
	return true
end

local function calc_distance(diff_mat, training_label)
	local distance = {}
	for i = 1, diff_mat:rows() do
		local dis = 0.0
		for j = 1, diff_mat:columns() do
			local element = diff_mat:getelement(i, j)
			dis = dis + (element * element)
			assert(dis >= 0)
		end
		table.insert(distance, {dis, training_label[i]})
	end

	table.sort(distance, function (t1, t2)
		return t1[1] < t2[1]
	end)
	return distance
end

local function comparison_label(distance, comparison_count)
	local label_count = {}
	for i = 1, comparison_count do
		local dis = distance[i]
		if not dis then
			break
		end
		local label = dis[2]
		label_count[label] = (label_count[label] or 0) + 1
	end

	local label, label_value
	for k, v in pairs(label_count) do
		if not label_value or v > label_value then
			label = k
			label_value = v
		end
	end	
	return label
end

function mt:classify(vector, comparison_count)
	if not self.training_mat then
		self.training_mat = matrix:new(self.training_data)
	end

	local diff_mat = matrix:new(self.training_mat:rows(), vector) - self.training_mat
	local distance = calc_distance(diff_mat, self.training_label)

	return comparison_label(distance, comparison_count)
end

return knn