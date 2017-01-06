package.path = package.path .. ";../?.lua" 

local tree = require "tree"
local bayes = require "bayes"

local function create_dict(dataset)
	local dict = {}
    local index = 0

    for _, datarow in ipairs(dataset) do
	    for _, element in ipairs(datarow) do
	    	if not dict[element] then
	    		index = index + 1
	    		dict[element] = index
	    	end
	    end
    end
    return dict, index
end

local function datarow2vector(dict, dict_count, datarow)
	local array = bayes.create_array(0, dict_count)
	for _, element in pairs(datarow) do
		local index = dict[element]
		if index then
			array[index] = array[index] + 1
		end
	end
	return array
end

local function main()
	local dataset ={
		{'my', 'dog', 'has', 'flea', 'problems', 'help', 'please'},
		{'maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'},
		{'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'},
		{'stop', 'posting', 'stupid', 'worthless', 'garbage'},
		{'mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'},
		{'quit', 'buying', 'worthless', 'dog', 'food', 'stupid'},
	}
	local labels = {'normal', 'abusive', 'normal', 'abusive', 'normal', 'abusive'}

    local dict, dict_count = create_dict(dataset)

	local key = "main_test"
	local ml = bayes.create()

    for index, datarow in ipairs(dataset) do
    	local label = labels[index]
    	local vector = datarow2vector(dict, dict_count, datarow)

		ml:add_training_data(label, vector)
    end  

	ml:training()

	local test_vect1 = datarow2vector(dict, dict_count, {'love', 'my', 'dalmation'})
	assert(ml:classify(test_vect1) == 'normal')

	local test_vect2 = datarow2vector(dict, dict_count, {'stupid', 'garbage'})
	assert(ml:classify(test_vect2) == 'abusive')
end

main()
