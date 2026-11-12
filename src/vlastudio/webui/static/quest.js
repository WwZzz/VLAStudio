"use strict";
(() => {
  const panel = $("questPanel"), root = $("questHands");
  const trails = new Map(), views = new Map();
  let session = "", signature = "", request, timer, animation, clock = 0, received = 0, generation = 0;
  function stop() {
    generation++;
    clearTimeout(timer);
    cancelAnimationFrame(animation);
    request?.abort();
    request = null;
  }
  function bindings(config, hand) {
    const right = hand === "right", open = right ? "A" : "X", home = right ? "B" : "Y";
    const analog = config.gripper_trigger_mode === "analog_travel";
    const normal = config.gripper_default_closed !== false;
    const inverse = !!config.gripper_invert;
    const releasedOpen = normal !== inverse;
    const trigger = !config.gripper_use_trigger ? "Unassigned" :
      analog && !config.gripper_delta_mode ?
        releasedOpen ? "Release to open · Pull to close" : "Release to close · Pull to open" :
        "Controls gripper while gripping";
    return [
      ["Grip", "Hold to follow · Release to hold"],
      ["Trigger", trigger],
      [open, config.gripper_a_maps_to_open !== false ? "Open gripper while gripping" : "Unassigned"],
      [home, "Reset robot"],
      ["Stick", config.publish_thumbsticks ? "Configured base control" : "Unassigned"],
      [right ? "Meta" : "Menu", "Headset system menu"],
    ];
  }
  function render() {
    const s = state?.session;
    const config = s?.quest?.config;
    const show = collecting && (config || profiles?.quest_profiles?.includes($("teleop").value));
    panel.hidden = !show;
    if (show) $("questUrl").textContent = profiles?.quest_url || tr("See the HTTPS address in the Quest service log.");
    $("questTrailHint").hidden = !config;
    if (!show || !s?.active || !config) {
      stop(); session = ""; signature = ""; trails.clear(); views.clear(); root.replaceChildren(); return;
    }
    const key = JSON.stringify([s.id, config, language]);
    if (key !== signature) {
      stop(); signature = key;
      if (session !== s.id) trails.clear();
      session = s.id; views.clear(); root.replaceChildren();
      const hands = config.arm_mode === "dual" || !config.arm_mode ? ["left", "right"] : [config.arm_mode];
      for (const hand of hands) {
        const card = el("div", "quest-hand"), heading = el("div", "heading");
        const status = el("small", "", tr("Waiting for tracking"));
        heading.append(el("h3", "", tr(hand === "left" ? "Left controller" : "Right controller")), status);
        const canvas = el("canvas", "quest-trail");
        canvas.setAttribute("role", "img");
        canvas.setAttribute("aria-label", tr("Controller trajectory"));
        const controls = el("dl", "quest-controls");
        const keys = new Map();
        for (const [button, description] of bindings(config, hand)) {
          const dt = el("dt"), cap = el("kbd", "", button);
          dt.append(cap); controls.append(dt, el("dd", "", tr(description))); keys.set(button, cap);
        }
        card.append(heading, canvas, controls); root.append(card);
        views.set(hand, {canvas, status, keys});
      }
      poll(); draw();
    }
  }
  async function poll() {
    if (!session || panel.hidden) return;
    const token = generation;
    if (!document.hidden && panel.open && !reviewingEpisode()) {
      request = new AbortController();
      const id = session;
      try {
        const r = await fetch("/api/quest-preview?id=" + encodeURIComponent(id), {signal:request.signal, cache:"no-store"});
        if (r.ok) {
          const data = await r.json();
          if (session !== id || token !== generation) return;
          clock = data.now; received = performance.now();
          for (const sample of data.samples) {
            if (!views.has(sample.hand)) continue;
            const trail = trails.get(sample.hand) || [];
            if (!trail.length || sample.stamp > trail.at(-1).stamp) trail.push(sample);
            while (trail.length > 100 || (trail.length && data.now - trail[0].stamp > 4)) trail.shift();
            trails.set(sample.hand, trail);
          }
        }
      } catch (error) {
        if (error.name !== "AbortError") clock = 0;
      }
    }
    if (session && token === generation) timer = setTimeout(poll, 80);
  }
  // Fixed camera projection of WebXR coordinates. X right, Y up, Z depth.
  const project = p => [p[0]*.85-p[2]*.55, -p[1]+p[2]*.28];
  function draw() {
    if (!session || panel.hidden) return;
    if (!document.hidden && panel.open && !reviewingEpisode()) {
      const now = clock + (performance.now()-received)/1000;
      for (const [hand, view] of views) {
        const {canvas, status, keys} = view, trail = trails.get(hand) || [];
        while (trail.length && now-trail[0].stamp>4) trail.shift();
        const w = canvas.clientWidth, h = 190, dpr = Math.min(devicePixelRatio || 1, 2);
        if (canvas.width !== Math.round(w*dpr) || canvas.height !== h*dpr) {
          canvas.width = Math.round(w*dpr); canvas.height = h*dpr;
        }
        const c = canvas.getContext("2d");
        c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,w,h);
        const last = trail.at(-1), fresh = last && now-last.stamp<.6 && clock;
        status.textContent = tr(!fresh ? "Waiting for tracking" : last.grip ? "Grip held" : "Grip released");
        for (const [key, cap] of keys) {
          const on = fresh && (key === "Grip" ? last.grip :
            key === "Trigger" ? last.trigger>.06 : !!last.buttons[key.toLowerCase()]);
          cap.classList.toggle("pressed", !!on);
        }
        c.strokeStyle = themeColor("--chart-grid"); c.lineWidth = 1;
        for (let x=16;x<w;x+=24) {c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke();}
        for (let y=10;y<h;y+=24) {c.beginPath();c.moveTo(0,y);c.lineTo(w,y);c.stroke();}
        if (!last) continue;
        // Keep a stable origin for each session. One metre uses 95 pixels.
        if (!view.origin) view.origin = project(last.position);
        const projected = project(last.position);
        for (let axis=0;axis<2;axis++) {
          const margin = (axis ? h : w)/2 - 28;
          const offset = (projected[axis]-view.origin[axis])*95;
          if (Math.abs(offset)>margin) view.origin[axis] += (offset-Math.sign(offset)*margin)/95;
        }
        const xy = p => {const a=project(p);return [w/2+(a[0]-view.origin[0])*95,h/2+(a[1]-view.origin[1])*95];};
        const color = hand === "left" ? "#78b6ef" : themeColor("--green");
        c.lineWidth = 2.5; c.strokeStyle = color;
        for (let i=1;i<trail.length;i++) {
          if (trail[i].stamp-trail[i-1].stamp>.3) continue;
          c.globalAlpha = Math.max(0,1-(now-trail[i].stamp)/4);
          c.beginPath();c.moveTo(...xy(trail[i-1].position));c.lineTo(...xy(trail[i].position));c.stroke();
        }
        c.globalAlpha = fresh ? 1 : Math.max(0,1-(now-last.stamp)/4);
        const [x,y] = xy(last.position);
        // Small controller body follows the measured quaternion's roll.
        const [qx,qy,qz,qw] = last.quaternion;
        c.save();c.translate(x,y);c.rotate(Math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)));
        c.fillStyle = color;c.beginPath();c.roundRect(-5,-10,10,20,4);c.fill();
        c.strokeStyle = color;c.lineWidth=2;c.beginPath();c.ellipse(0,-11,10,6,0,0,Math.PI*2);c.stroke();c.restore();
        c.globalAlpha=1;
        c.font="11px monospace";c.fillStyle=themeColor("--muted");
        c.fillText("X →   Y ↑   Z ↙", 10,h-10);
      }
    }
    animation = requestAnimationFrame(draw);
  }
  document.addEventListener("episode-session", render);
  $("teleop").addEventListener("change", render);
  panel.addEventListener("toggle", () => {if (!panel.open) return; render();});
})();
