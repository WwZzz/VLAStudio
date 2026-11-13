"use strict";
class DeviceConfigEditor {
  constructor(root) { this.root = root; this.devices = []; this.nextId = 0; }
  load(document, types) {
    this.root.replaceChildren(); this.devices = [];
    this.asList = Array.isArray(document);
    this.types = types;
    for (const row of this.asList ? document : [document]) this.add(row);
  }
  updateTypes(types) {
    this.types = types;
    for (const list of this.root.querySelectorAll("datalist"))
      list.replaceChildren(...types.map(type=>new Option(type,type)));
  }
  field(key, value) {
    const row = el("div", "device-parameter"), name = el("input", "");
    name.value = key; name.ariaLabel = "parameter name"; name.spellcheck = false;
    const kind = el("select", ""), input = el("textarea", "");
    input.rows = 1; input.spellcheck = false; input.ariaLabel = key || "parameter value";
    for (const t of ["string", "number", "boolean", "null", "array", "object"]) kind.add(new Option(t,t));
    kind.value = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
    input.value = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    input.rows = Math.min(7, input.value.split("\n").length);
    kind.ariaLabel = "parameter type";
    const remove = el("button", "device-remove", "×");
    remove.type = "button"; remove.title = tr("Remove parameter");
    remove.onclick = () => row.remove();
    row.append(name, kind, input, remove);
    row.value = () => {
      const key = name.value.trim();
      if (!key) throw Error(tr("Parameter name is required"));
      const type = kind.value;
      let value;
      try { value = type === "string" ? input.value : JSON.parse(input.value); }
      catch { throw Error(key + " — " + tr("Enter a valid value for the selected type")); }
      const actual = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
      if (type !== actual) throw Error(key + " — " + tr("Value does not match its type"));
      return [key, value];
    };
    return row;
  }
  values(list) {
    const result = {};
    for (const row of list.children) {
      const [key, value] = row.value();
      if (Object.hasOwn(result,key)) throw Error(tr("Duplicate parameter") + " " + key);
      Object.defineProperty(result,key,{value,enumerable:true,writable:true,configurable:true});
    }
    return result;
  }
  add(row = {type:this.types.find(t => t.endsWith(".OpenCVCamera")) || this.types[0], name:"device", args:{}}) {
    const card = el("details", "device-card"), summary = el("summary", "");
    card.open = true;
    const title = el("strong", "", row.name || row.args?.name || row.type.split(".").at(-1));
    const remove = el("button", "device-remove", tr("Remove device")); remove.type = "button";
    remove.onclick = event => {event.preventDefault(); card.remove();};
    summary.append(title, remove); card.append(summary);
    const top = el("div", "device-identity"), type = el("input", ""), name = el("input", "");
    type.value = row.type; type.spellcheck = false; type.ariaLabel = "type";
    const datalist = el("datalist", ""); datalist.id = "deviceTypes-" + (++this.nextId);
    for (const t of this.types) datalist.append(new Option(t,t));
    type.setAttribute("list",datalist.id);
    name.value = row.name || ""; name.ariaLabel = "name"; name.spellcheck = false;
    for (const [key, input] of [["type",type],["name",name]]) {
      const label = el("label", "", key); label.append(input); top.append(label);
    }
    name.oninput = () => title.textContent = name.value || type.value.split(".").at(-1);
    card.append(top, datalist, el("h4", "", "args"));
    const fields = el("div", "device-parameters");
    for (const [key,value] of Object.entries(row.args || {})) fields.append(this.field(key,value));
    const actions = el("div", "device-actions"), add = el("button", "", tr("Add parameter"));
    const defaults = el("button", "", tr("Fill defaults")), message = el("p", "");
    add.type = defaults.type = "button";
    add.onclick = () => { const field=this.field("", "");fields.append(field);field.firstElementChild.focus(); };
    defaults.onclick = async () => {
      defaults.disabled = true; message.textContent = "";
      const reference = type.value.trim();
      try {
        const schema = await api("device-parameters?" + new URLSearchParams({type:reference}));
        if (!card.isConnected || type.value.trim() !== reference) return;
        const existing = new Set(Array.from(fields.children, child=>child.firstElementChild.value));
        for (const [key, spec] of Object.entries(schema)) {
          if (!existing.has(key)) {
            const field = this.field(key, spec.required ? "" : spec.default);
            field.firstElementChild.title = spec.required ? "required" : "default";
            fields.append(field);
          }
        }
      } catch (e) { message.textContent = e.message; }
      finally { defaults.disabled = false; }
    };
    actions.append(add, defaults);card.append(fields,actions,message);
    const extras = Object.fromEntries(Object.entries(row).filter(([k])=>!["type","name","args"].includes(k)));
    const other = el("details","device-extras"), otherTitle=el("summary","",tr("Additional configuration fields"));
    const extraFields=el("div","device-parameters");
    for (const [key,value] of Object.entries(extras)) extraFields.append(this.field(key,value));
    const extraAdd=el("button","",tr("Add parameter"));extraAdd.type="button";
    extraAdd.onclick=()=>extraFields.append(this.field("",""));
    other.append(otherTitle,extraFields,extraAdd);card.append(other);
    card.value = () => {
      const result = this.values(extraFields);
      result.type = type.value.trim();
      if (name.value.trim()) result.name = name.value.trim();
      result.args = this.values(fields);
      if (Object.hasOwn(result.args, "name") && !result.args.name && result.name) result.args.name = result.name;
      return result;
    };
    this.root.append(card); this.devices.push(card);
    return card;
  }
  addCopy(row) {
    const names=new Set();
    for (const card of this.devices.filter(d=>d.isConnected)) {
      const name=card.querySelector('input[aria-label="name"]').value;
      if (name) names.add(name);
      for (const field of card.querySelector(".device-parameters").children) {
        if (field.firstElementChild.value==="name" || field.firstElementChild.value==="robot_id") names.add(field.querySelector("textarea").value);
      }
    }
    const base=row.args?.name || row.args?.robot_id || row.name || "device";
    let name=base, suffix=2;
    while (names.has(name)) name=base+"_"+suffix++;
    if (name!==base) {
      row.name=name;
      if (row.args?.name) row.args.name=name;
      else if (row.args?.robot_id) row.args.robot_id=name;
    }
    return this.add(row);
  }
  value() {
    return {devices:this.devices.filter(card=>card.isConnected).map(card=>card.value()), as_list:this.asList};
  }
}
