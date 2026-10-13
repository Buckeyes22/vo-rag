-- Keep EPUB anchor identifiers through the GFM writer (VOR-1427).
--
-- GFM has no identifier syntax, so pandoc drops the empty span it puts at the
-- start of every spine file ([]{#chapter01.xhtml}) and every source id
-- ([]{#chapter01.xhtml#p26}, headers, divs). Those ids are the only link
-- between the EPUB's table of contents and the body when chapter titles are
-- styled <p> paragraphs. This filter writes each id as a standalone paragraph
-- "⟦vorag-anchor:<id>⟧" before the block that carries it;
-- epub_markdown.markdown_to_elements turns matching ones into TITLE elements
-- and drops the rest. Ids are percent-encoded so the writer cannot escape them.
--
-- It also rewrites tables GFM cannot express so their text survives (see
-- rewrite_table below).
--
-- Only filter APIs available in pandoc 3.1 (production runs 3.1.3).

local function sentinel(id)
  -- Everything but [A-Za-z0-9.-#:/] is encoded, '_' included, so the writer's
  -- emphasis escaping can never touch the id.
  local encoded = id:gsub("[^%w%.%-#:/]", function(c)
    return string.format("%%%02X", string.byte(c))
  end)
  return pandoc.Para({ pandoc.Str("⟦vorag-anchor:" .. encoded .. "⟧") })
end

-- Ids of Spans and Links inside a list of inlines, in document order.
local function inline_ids(inlines)
  local ids = {}
  local function visit(list)
    for _, el in ipairs(list) do
      if (el.t == "Span" or el.t == "Link") and el.identifier ~= "" then
        ids[#ids + 1] = el.identifier
      end
      if el.content ~= nil and type(el.content) ~= "string" and el.t ~= "Note" then
        visit(el.content)
      end
    end
  end
  visit(inlines)
  return ids
end

-- True when the inlines render nothing: only empty spans and whitespace.
local function only_anchors(inlines)
  for _, el in ipairs(inlines) do
    if el.t == "Span" then
      if not only_anchors(el.content) then
        return false
      end
    elseif el.t ~= "Space" and el.t ~= "SoftBreak" and el.t ~= "LineBreak" then
      return false
    end
  end
  return true
end

local process

local function append_all(out, list)
  for _, b in ipairs(list) do
    out[#out + 1] = b
  end
end

-- Top-level blocks and Div contents only: a sentinel paragraph inside a list
-- item or table cell would change how the writer renders that structure.
process = function(blocks)
  local out = {}
  for _, block in ipairs(blocks) do
    local t = block.t
    if t == "Para" or t == "Plain" then
      for _, id in ipairs(inline_ids(block.content)) do
        out[#out + 1] = sentinel(id)
      end
      if not only_anchors(block.content) then
        out[#out + 1] = block
      end
    elseif t == "Header" then
      if block.identifier ~= "" then
        out[#out + 1] = sentinel(block.identifier)
      end
      for _, id in ipairs(inline_ids(block.content)) do
        out[#out + 1] = sentinel(id)
      end
      out[#out + 1] = block
    elseif t == "Div" then
      if block.identifier ~= "" then
        out[#out + 1] = sentinel(block.identifier)
      end
      -- Unwrap: the GFM writer renders a Div as its contents anyway.
      append_all(out, process(block.content))
    else
      out[#out + 1] = block
    end
  end
  return out
end

-- Tables the GFM writer cannot express (VOR-1427). With raw_html off, pandoc
-- replaces a table whose cells hold anything but one line of inlines with the
-- literal line "[TABLE]" and drops its text; books that lay out every page as
-- nested tables parsed to nothing but "[TABLE]" lines.
--   * A top-level table the writer can render as a pipe table is untouched.
--   * A data table -- at least two rows with two or more non-empty cells, and
--     every cell one line of inlines once Divs are unwrapped and line breaks
--     become spaces -- is rewritten that way and stays one pipe table.
--   * Anything else is a layout table: its caption and cells are unwrapped
--     into the parent in reading order (row by row, cell by cell). Table, row
--     and cell ids become Divs so process() writes their anchor sentinels.
-- A table nested in another table's cell was never written before (its parent
-- became "[TABLE]"), so it gets the data/layout test even when it is simple:
-- page-layout books nest spacer-column tables that would otherwise come out
-- as one padded pipe table per paragraph.

-- The gfm writer's own test (Writers.Shared.onlySimpleTableCells): a cell is
-- empty or a single Plain/Para with no LineBreak at any depth (the writer
-- queries the whole cell, so <b>a<br/>b</b> fails too).
local function simple_cell(blocks)
  if #blocks == 0 then
    return true
  end
  if #blocks > 1 or (blocks[1].t ~= "Plain" and blocks[1].t ~= "Para") then
    return false
  end
  local line_break = false
  pandoc.Blocks(blocks):walk({
    LineBreak = function(el)
      line_break = true
      return el
    end,
  })
  return not line_break
end

-- All rows of a table in reading order: head, each body's head and body rows,
-- then foot.
local function table_rows(tbl)
  local rows = {}
  local function add(list)
    for _, row in ipairs(list) do
      rows[#rows + 1] = row
    end
  end
  add(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do
    add(body.head)
    add(body.body)
  end
  add(tbl.foot.rows)
  return rows
end

-- The one line of inlines a cell holds once Divs are unwrapped, or nil when
-- it holds more than one Plain/Para or any other block.
local function cell_line(blocks)
  local found = nil
  local function visit(list)
    for _, b in ipairs(list) do
      if b.t == "Div" then
        if not visit(b.content) then
          return false
        end
      elseif (b.t == "Plain" or b.t == "Para") and found == nil then
        found = b.content
      else
        return false
      end
    end
    return true
  end
  if not visit(blocks) then
    return nil
  end
  if found == nil then
    return pandoc.Inlines({})
  end
  local line = pandoc.Plain(found):walk({
    LineBreak = function()
      return pandoc.Space()
    end,
  })
  return line.content
end

local function with_id(attr, blocks)
  if attr.identifier ~= "" then
    return { pandoc.Div(blocks, pandoc.Attr(attr.identifier)) }
  end
  return blocks
end

-- True when a cell shows anything: text other than (non-breaking) spaces, or
-- an image.
local function has_content(blocks)
  local text = pandoc.utils.stringify(blocks):gsub("\194\160", ""):gsub("%s", "")
  if text ~= "" then
    return true
  end
  local image = false
  pandoc.Blocks(blocks):walk({
    Image = function(el)
      image = true
      return el
    end,
  })
  return image
end

-- At least two rows showing two or more filled cells, counted on the grid: a
-- cell with a rowspan also fills the rows below it (their source rows are
-- shorter), and a colspan cell counts once.
local function looks_like_data(rows)
  local wide_rows = 0
  -- pending[col] = { rows still covered, filled } for rowspans from above.
  local pending = {}
  for _, row in ipairs(rows) do
    local filled = 0
    local col = 1
    local function drain_spans()
      while pending[col] ~= nil do
        local span = pending[col]
        if span.filled and span.first then
          filled = filled + 1
        end
        span.left = span.left - 1
        if span.left == 0 then
          pending[col] = nil
        end
        col = col + 1
      end
    end
    for _, cell in ipairs(row.cells) do
      drain_spans()
      local has = has_content(cell.contents)
      if has then
        filled = filled + 1
      end
      local width = math.max(cell.col_span or 1, 1)
      local height = math.max(cell.row_span or 1, 1)
      for k = 0, width - 1 do
        if height > 1 then
          pending[col + k] = { left = height - 1, filled = has, first = k == 0 }
        end
      end
      col = col + width
    end
    -- Spans from above to the right of this row's last cell.
    local max_col = col
    for c, _ in pairs(pending) do
      if c > max_col then
        max_col = c
      end
    end
    while col <= max_col do
      if pending[col] ~= nil then
        drain_spans()
      else
        col = col + 1
      end
    end
    if filled >= 2 then
      wide_rows = wide_rows + 1
    end
  end
  return wide_rows >= 2
end

-- Every id on a table, its rows and cells, and any Div, Header, Span or Link
-- inside the cells, in reading order. Collected before cells are flattened
-- (which unwraps their Divs).
local function table_ids(tbl, rows)
  local ids = {}
  local function add(id)
    if id ~= "" then
      ids[#ids + 1] = id
    end
  end
  local function collect(el)
    add(el.identifier)
    return nil
  end
  add(tbl.attr.identifier)
  for _, row in ipairs(rows) do
    add(row.attr.identifier)
    for _, cell in ipairs(row.cells) do
      add(cell.attr.identifier)
      pandoc.Blocks(cell.contents):walk({
        traverse = "topdown",
        Div = collect,
        Header = collect,
        Span = collect,
        Link = collect,
      })
    end
  end
  return ids
end

-- A table that stays a table, preceded by an empty Div per id so process()
-- writes their sentinels: the GFM table has no place for ids, so a TOC entry
-- pointing at a table or into its cells would otherwise never fire.
local function kept_table(tbl, ids)
  if #ids == 0 then
    return tbl
  end
  local out = {}
  for _, id in ipairs(ids) do
    out[#out + 1] = pandoc.Div({}, pandoc.Attr(id))
  end
  out[#out + 1] = tbl
  return out
end

local rewrite_table

local function rewrite_nested(blocks)
  return pandoc.Blocks(blocks):walk({
    traverse = "topdown",
    Table = function(tbl)
      return rewrite_table(tbl, true), false
    end,
  })
end

rewrite_table = function(tbl, nested)
  local rows = table_rows(tbl)
  for _, row in ipairs(rows) do
    for _, cell in ipairs(row.cells) do
      cell.contents = rewrite_nested(cell.contents)
    end
  end

  local ids = table_ids(tbl, rows)
  local all_simple = true
  for _, row in ipairs(rows) do
    for _, cell in ipairs(row.cells) do
      if not simple_cell(cell.contents) then
        all_simple = false
      end
    end
  end
  if all_simple and not nested then
    return kept_table(tbl, ids)
  end

  if looks_like_data(rows) then
    local lines = {}
    for _, row in ipairs(rows) do
      for _, cell in ipairs(row.cells) do
        local line = cell_line(cell.contents)
        if line == nil then
          lines = nil
          break
        end
        lines[#lines + 1] = { cell, line }
      end
      if lines == nil then
        break
      end
    end
    if lines ~= nil then
      for _, pair in ipairs(lines) do
        local cell, line = pair[1], pair[2]
        if not simple_cell(cell.contents) then
          cell.contents = #line > 0 and { pandoc.Plain(line) } or {}
        end
      end
      return kept_table(tbl, ids)
    end
  end

  local out = {}
  if tbl.caption and tbl.caption.long then
    append_all(out, tbl.caption.long)
  end
  for _, row in ipairs(rows) do
    local row_blocks = {}
    for _, cell in ipairs(row.cells) do
      append_all(row_blocks, with_id(cell.attr, cell.contents))
    end
    append_all(out, with_id(row.attr, row_blocks))
  end
  return with_id(tbl.attr, out)
end

function Pandoc(doc)
  doc = doc:walk({
    traverse = "topdown",
    Table = function(tbl)
      return rewrite_table(tbl, false), false
    end,
  })
  return pandoc.Pandoc(process(doc.blocks), doc.meta)
end
