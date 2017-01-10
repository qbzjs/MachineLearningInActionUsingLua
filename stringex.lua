local function string_split(str, sep)
	local sep = string.format("([^%s]+)", sep or "%s")
    local tab = {}

    for sub in string.gmatch(str, sep) do
        table.insert(tab, sub)
    end
    
    return tab
end

_G.string.split = string_split