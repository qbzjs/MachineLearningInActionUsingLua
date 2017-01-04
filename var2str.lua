local function StrVal2Str(szVal)
	szVal	= string.gsub(szVal, "\\", "\\\\");
	szVal	= string.gsub(szVal, '"', '\\"');
	szVal	= string.gsub(szVal, "\n", "\\n");
	szVal	= string.gsub(szVal, "\r", "\\r");
	--szVal	= string.format("%q", szVal);
	return '"'..szVal..'"';
end;

local function Val2Str(var, szBlank)
	local szType	= type(var);
	if (szType == "nil") then
		return "nil";
	elseif (szType == "number") then
		return tostring(var);
	elseif (szType == "string") then
		return StrVal2Str(var);
	elseif (szType == "function") then
		local szCode	= string.dump(var);
		local arByte	= {string.byte(szCode, i, #szCode)};
		szCode	= "";
		for i = 1, #arByte do
			szCode	= szCode..'\\'..arByte[i];
		end;
		return 'loadstring("' .. szCode .. '")';
	elseif (szType == "table") then
		if not szBlank then
			szBlank	= "";
		end;
		local szTbBlank	= szBlank .. "  ";
		local szCode	= "";
		for k, v in pairs(var) do
			local szPair	= szTbBlank.."[" .. Val2Str(k) .. "]	= " .. Val2Str(v, szTbBlank) .. ",\n";
			szCode	= szCode .. szPair;
		end;
		if (szCode == "") then
			return "{}";
		else
			return "\n"..szBlank.."{\n"..szCode..szBlank.."}";
		end;
	elseif szType == "boolean" then
		return var and "true" or "false";
	else	--if (szType == "userdata") then
		return '"' .. tostring(var) .. '"';
	end;
end;

return Val2Str;
