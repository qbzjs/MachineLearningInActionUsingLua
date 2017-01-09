local var2str = require "var2str"
local tree = require "tree"
local matrix = require "matrix"

local logistic_regression = {}
local mt = {}

function logistic_regression.create()
	local ml = {}
	ml.training_label = {}
	ml.training_data = {}

	setmetatable(ml, {__index= mt})
	return ml
end

function mt:add_training_data(label, vector)
	assert(type(vector) == "table")

	table.insert(self.training_label, label)
	table.insert(self.training_data, vector)
end

local function sigmoid(in_x)
	return 1.0 / (1 + math.exp(-in_x))
end

local function sigmoid_mat(mat)
	for i = 1, mat:rows() do
		for j = 1, mat:columns() do
			local element = mat:getelement(i, j)
			mat:setelement(i, j, sigmoid(element))
		end
	end
end

function mt:training()		
    local alpha = 0.001
    local max_cycles = 500

	if not self.training_data_mat then
		self.training_data_mat = matrix:new(self.training_data)
	end

	if not self.training_label_mat then
		self.training_label_mat = matrix:new({self.training_label}):transpose()
	end

    local training_data_mat_transpose = self.training_data_mat:transpose()

	local weights = matrix:new(self.training_data_mat:columns(), 1, 1)
    for i = 1, max_cycles do
    	local h = self.training_data_mat * weights
    	sigmoid_mat(h)
    	
    	local err = self.training_label_mat - h
    	err = training_data_mat_transpose * err 
    	err = err * alpha
        weights = weights + err
	end	

	self.weights = weights
end

function mt:classify(vector)
	local value = matrix:new(vector) * self.weights
	local prob = sigmoid(value:getelement(1, 1))
	return prob > 0.5 and 1 or 0
end

function mt:save(path)
	local file = assert(io.open(path, "w"))
	file:write(string.format("local weights = %s;\n", var2str(self.weights)))
	file:write("return weights")
	file:close()
end

function mt:load(path)
	local file = io.open(path, "r")
	if not file then
		return false
	end
	local content = file:read("a")
	local fn = load(content)
	local weights = fn()
	self.weights = weights
	return true
end

return logistic_regression
