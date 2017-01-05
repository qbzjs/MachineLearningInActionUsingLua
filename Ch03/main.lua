package.path = package.path .. ";../?.lua" 

local tree = require "tree"
local decision_tree = require "decision_tree"

local function main()
	math.randomseed(os.clock())

	local key = "main_test"
	local ml = decision_tree.create()

	if not ml:load(key) then
		ml:add_training_data('yes', {1, 1})
		ml:add_training_data('yes', {1, 1})
		ml:add_training_data('no', {1, 0})
		ml:add_training_data('no', {0, 1})
		ml:add_training_data('no', {0, 1})

		ml:training()
		ml:save(key)
		tree("ml", ml)
	end

	assert('yes' == ml:classify({1, 1}))
	assert('no' == ml:classify({0, 1}))
end

main()
