local var2str = require "var2str"
local tree = require "tree"

local bayes = {}
local mt = {}

function bayes.create()
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

local function create_array(value, count)
	local array = {}
	for i = 1, count do
		array[i] = value
	end
	return array
end

local function denom_array(array)
	local sum = 0
	for i = 1, #array do
		sum = sum + array[i]
	end
	return sum
end

local function add_array(array1, array2)
	assert(#array1 == #array2)

	local array = {}
	for i = 1, #array1 do
		array[i] = array1[i] + array2[i]
	end
	return array
end

local function mul_array(array1, array2)
	assert(#array1 == #array2)

	local array = {}
	for i = 1, #array1 do
		array[i] = array1[i] * array2[i]
	end
	return array
end

local function div_array_num(array, num)
	local new_array = {}
	for i = 1, #array do
		new_array[i] = array[i] / num
	end
	return new_array
end

local function log_array_num(array, num)
	local new_array = {}
	for i = 1, #array do
		new_array[i] = math.log(array[i], num)
	end
	return new_array
end

function mt:training()
	local columns_count = #(self.training_data[1])

	self.probability = {}
	for i, v in ipairs(self.training_data) do
		local label = self.training_label[i]
		local prob = self.probability[label]
		if not prob then
			prob = {
				vector = create_array(1, columns_count),
				denom = 2,
				label_count = 0,
			}
			self.probability[label] = prob
		end

		local vector = self.training_data[i]
		prob.vector = add_array(prob.vector, vector)
		prob.denom = prob.denom + denom_array(vector)
		prob.label_count = prob.label_count + 1
	end

	local rows_count = #self.training_data
	for label, prob in pairs(self.probability) do
		prob.prob_vector = div_array_num(prob.vector, prob.denom)
		prob.prob_vector = log_array_num(prob.prob_vector)

		prob.label_prob = prob.label_count / rows_count
	end
end

function mt:classify(vector)
	local labels_prob = {}
	for label, prob in pairs(self.probability) do
		local prob_vector = mul_array(vector, prob.prob_vector)
		labels_prob[label] = denom_array(prob_vector) + math.log(prob.label_prob)
	end

	local max_prob = nil
	local max_prob_label = nil

	for label, prob in pairs(labels_prob) do
		if not max_prob or prob > max_prob then
			max_prob = prob
			max_prob_label = label
		end
	end
	return max_prob_label
end

function mt:save(path)
	local file = assert(io.open(path, "w"))
	file:write(string.format("local probability = %s;\n", var2str(self.probability)))
	file:write("return probability")
	file:close()
end

function mt:load(path)
	local file = io.open(path, "r")
	if not file then
		return false
	end
	local content = file:read("a")
	local fn = load(content)
	local probability = fn()
	self.probability = probability
	return true
end

bayes.create_array = create_array
return bayes
