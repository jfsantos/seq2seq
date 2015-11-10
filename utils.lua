require 'nngraph';

function customToDot(graph, title, failedNode)
   local str = graph:todot(title)
   if not failedNode then
      return str
   end

   local failedNodeId = nil
   for i, node in ipairs(graph.nodes) do
      if node.data == failedNode.data then
         failedNodeId = node.id
         break
      end
   end

   if failedNodeId ~= nil then
      -- The closing '}' is removed.
      -- And red fillcolor is specified for the failedNode.
      str = string.gsub(str, '}%s*$', '')
      str = str .. string.format('n%s[style=filled, fillcolor=red];\n}',
      failedNodeId)
   end
   return str
end

function saveSvg(svgPathPrefix, dotStr)
   io.stderr:write(string.format("saving %s.svg\n", svgPathPrefix))
   local dotPath = svgPathPrefix .. '.dot'
   local dotFile = io.open(dotPath, 'w')
   dotFile:write(dotStr)
   dotFile:close()

   local svgPath = svgPathPrefix .. '.svg'
   local cmd = string.format('dot -Tsvg -o %s %s', svgPath, dotPath)
   os.execute(cmd)
end

function outputGraphViz(gmodule, focusNode)
   local focusNode = focusNode or gmodule.outnode
   local nInputs = gmodule.nInputs or #gmodule.innode.children
   local svgPathPrefix = gmodule.name or string.format(
   'nngraph_%sin_%sout', nInputs, #gmodule.outnode.children)
   local dotStr = customToDot(gmodule.fg, svgPathPrefix, focusNode)
   saveSvg(svgPathPrefix, dotStr)
end
