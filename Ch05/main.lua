package.path = package.path .. ";../?.lua" 

local tree = require "stringex"
local tree = require "tree"
local logistic_regression = require "logistic_regression"

local function load_dataset(ml)
	local file = assert(io.open("testSet.txt", "r"))
	local vector = {}
	for line in file:lines("l") do 
		local str = string.split(line)
		assert(#str == 3)
		local vector = {tonumber(str[1]), tonumber(str[2])}
		local label = tonumber(str[3])

		ml:add_training_data(label, vector)
	end
	file:close()
end

local function main()
	local key = "main_test"
	local ml = logistic_regression.create()

	if not ml:load(key) then
		load_dataset(ml)
		ml:training()
		ml:save(key)
	end	
end

main()
