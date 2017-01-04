package.path = package.path .. ";../?.lua" 

local tree = require "tree"
local knn = require "knn"

local function img2vector(path)
	local file = io.open(path, "r")
	if not file then
		return
	end

	local vector = {}
	for line in file:lines("l") do 
		local num = tonumber(line, 2)
		table.insert(vector, num)
	end
	file:close()
	return vector
end

local function load_img(dir, num, index)
	local path = string.format("%s/%d_%d.txt", dir, num, index)
	local vector = img2vector(path)
	return vector
end

local function load_training(dir, ml)
	for num = 0, 9 do
		local index = 0
		while true do
			local vector = load_img(dir, num, index)
			if not vector then
				break
			end

			index = index + 1
			ml:add_training(num, vector)
		end
	end	
end

local function test(dir, ml, k)
	local total_count = 0
	local error_count = 0

	for num = 0, 9 do
		local index = 0
		while true do
			local vector = load_img(dir, num, index)
			if not vector then
				break
			end
			index = index + 1

			local result = ml:classify(vector, k)

			total_count = total_count + 1
			if result ~= num then
				error_count = error_count + 1
			end
		end
	end	

	return total_count, error_count
end

local function main()
	local key = "main_training"
	local ml = knn.create()
	if not ml:load(key) then
		load_training("digits/trainingDigits", ml)
		ml:save(key)
	end
	
	local total_count, error_count = test("digits/testDigits", ml, 4)
	print(string.format("total count:%d, error_count:%d, error_rate:%f", total_count, error_count, error_count / total_count))
end

local function main_test()
	math.randomseed(os.clock())

	local key = "main_test"
	local ml = knn.create()

	if not ml:load(key) then
		for i = 1, 5 do
			ml:add_training(0, {math.random(1, 100)})
		end

		for i = 1, 5 do
			ml:add_training(1, {math.random(101, 200)})
		end
		ml:save(key)
		tree(ml)
	end

	assert(0 == ml:classify({20}, 5))
	assert(1 == ml:classify({180}, 5))
end

main()
