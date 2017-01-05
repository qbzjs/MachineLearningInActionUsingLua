local var2str = require "var2str"

local function count_column_values(dataset, axis)
	local value_counts = {}
	local total_count = 0
	for index, row in ipairs(dataset) do
		local value = row[axis]
		if not value_counts[value] then
			value_counts[value] = 0
			total_count = total_count + 1
		end
		value_counts[value] = value_counts[value] + 1
	end
	return value_counts, total_count
end

local function calc_shannon_ent(dataset, laber_index)
	local laber_counts = count_column_values(dataset, laber_index)
	local data_counts = #dataset

	local shannon_ent = 0.0
	for label, counts in pairs(laber_counts) do
		local prob = counts / data_counts
		shannon_ent = shannon_ent - prob * math.log(prob, 2)
	end
    return shannon_ent
end

local function split_dataset(dataset, axis, value)
	local new_dataset = {}
	for index, row in ipairs(dataset) do
		if row[axis] == value then
			local new_row = {table.unpack(row)}
			table.remove(new_row, axis)
			table.insert(new_dataset, new_row)
		end
	end 
	return new_dataset
end

local function choose_best_feature_to_split(dataset)
	local column_count = #(dataset[1])
	local base_entropy = calc_shannon_ent(dataset, column_count)
	local best_info_gain = 0.0
	local best_feature = nil
	local best_feat_list = nil
	local features_count = column_count - 1

	for i = 1, features_count do
		local feat_list = count_column_values(dataset, i)
		local new_entropy = 0.0
		for value, count in pairs(feat_list) do
			local sub_dataset = split_dataset(dataset, i, value)
            local prob = #sub_dataset / #dataset
            new_entropy = new_entropy + prob * calc_shannon_ent(sub_dataset, #(sub_dataset[1]))  
		end

		local info_gain = base_entropy - new_entropy
		if info_gain > best_info_gain then
			best_info_gain = info_gain  
            best_feature = i
            best_feat_list = feat_list
		end
	end

	return best_feature, best_feat_list
end

local function create_tree_label(dataset)
	local column_count = #(dataset[1])
	local laber_counts, total_count = count_column_values(dataset, column_count)
	if column_count ~= 1 and total_count ~= 1 then
		return
	end

	local majority_label = nil
	local majority_counts = 0
	for label, counts in pairs(laber_counts) do
		if counts > majority_counts then
			majority_label = label
			majority_counts = counts
		end
	end

	return majority_label
end

local function create_tree(dataset)
	local majority_label = create_tree_label(dataset)
	if majority_label then
		return {label = majority_label}
	end

	local best_feat, best_feat_list = choose_best_feature_to_split(dataset)
	local nodes = {}
	for value, count in pairs(best_feat_list) do
		local sub_dataset = split_dataset(dataset, best_feat, value) 
		nodes[value] = create_tree(sub_dataset)
	end

	return {feat_index = best_feat, feat_nodes = nodes}
end

local function classify(input_tree, vector)
	local value = vector[input_tree.feat_index]
	local sub_tree = input_tree.feat_nodes[value]
	if not sub_tree then
		return
	end

	if sub_tree.feat_index then
		return classify(sub_tree, vector)
	end

	assert(sub_tree.label)
	return sub_tree.label
end

local decision_tree = {}
local mt = {}

function decision_tree.create()
	local ml = {}
	ml.training_data = {}

	setmetatable(ml, {__index= mt})
	return ml
end

function mt:add_training_data(label, vector)
	assert(type(vector) == "table")

	table.insert(vector, label)
	table.insert(self.training_data, vector)
end

function mt:training()
	self.decision_tree = create_tree(self.training_data)
end

function mt:save(path)
	local file = assert(io.open(path, "w"))
	file:write(string.format("local decision_tree = %s;\n", var2str(self.decision_tree)))
	file:write("return decision_tree")
	file:close()
end

function mt:load(path)
	local file = io.open(path, "r")
	if not file then
		return false
	end
	local content = file:read("a")
	local fn = load(content)
	local decision_tree = fn()
	self.decision_tree = decision_tree
	return true
end

function mt:classify(vector)
	return classify(self.decision_tree, vector)
end

return decision_tree
