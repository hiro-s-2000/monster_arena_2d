"""Gen4 attack strikes. The body no longer spins flat.

User request on 2026-11-04: 8-direction attacks must not rotate the
whole sprite onto its side. Frames 5-12 drive the limbs, jaws, wings
and tails the sheet already has. Idle, walk and hurt are untouched.
A peak never exceeds an angle that script already draws, except a
medium wind-up (8 to 22 degrees) which is opened to a readable hit.
"""

from pathlib import Path

# Frames 0-4 stay the authored roar. The strike is already near full on
# frame 5 so a raised cannon or jaw does not dip, then it holds through
# the impact and settles. No whole-body lunge: positive dx is forward on
# some species and backward on left-facing ones.
ENV = (
    0, 0, 0, 0, 0,
    0.88, 1.0, 1.0, 1.0, 0.96, 0.88,
    0.72, 0.48,
    0.18, 0.06, 0,
)

PEAKS = {
    "forge_abomasnow_actions": {"arm_l": 24.3, "arm_r": -24.3},
    "forge_abomasnow_mega_actions": {"arm_l": 22.0, "arm_r": -24.0, "jaw": 3.0, "tree": 5.0},
    "forge_bastiodon_actions": {"jaw": 3.0},
    "forge_bibarel_actions": {"ldy": 4.0, "rdy": 4.0, "tail": 6.0},
    "forge_bidoof_actions": {"jaw": 2.0, "tail": 22.0},
    "forge_bonsly_actions": {"open": 18.0},
    "forge_bronzong_actions": {"arm": 16.0},
    "forge_budew_actions": {"open": 16.0},
    "forge_buizel_actions": {"inner": 22.0, "outer": 22.0},
    "forge_buneary_actions": {"ear_roll": 5.0, "ear_up": 22.0, "jaw": 2.0},
    "forge_burmy_actions": {"flare": 18.0},
    "forge_carnivine_actions": {"jaw": 4.0, "left": 22.0, "right": -22.0, "stem": 6.0},
    "forge_chatot_actions": {"foot_l": 30.0, "foot_r": -30.0, "jaw": 32.0, "note": 5.0, "tail": -7.0},
    "forge_cherubi_actions": {"leaf": -8.0},
    "forge_chimchar_actions": {"arm": 24.3, "flame_ang": 22.0, "foot_l": 28.0, "foot_r": 28.0, "mouth": 3.0},
    "forge_cherrim_actions": {"open": 14.0},
    "forge_chingling_actions": {"curl_l": 18.0, "curl_r": 20.0},
    "forge_combee_actions": {"ant": 7.0, "wing": 22.0},
    "forge_cranidos_actions": {"arm": 22.0, "foot_l": 24.0, "foot_r": -24.0, "mouth": 2.0, "tail": 7.0},
    "forge_croagunk_actions": {"arm_l": 22.0, "arm_r": -22.0},
    "forge_darkrai_mega_actions": {"cloth_l": 22.0, "cloth_r": 22.0, "cloth_tail": 22.0, "flame": 32.0, "hem": 6.0},
    "forge_drapion_actions": {"jaw": 3.0, "lclaw": -22.0, "rclaw": 22.0, "sting": 22.0},
    "forge_drifblim_actions": {"sway": 8.0},
    "forge_drifloon_actions": {"sway": 5.0},
    "forge_electivire_actions": {"mouth": 4.0},
    "forge_empoleon_actions": {"flip_l": 22.0, "flip_r": -22.0, "jaw": 2.0},
    "forge_finneon_actions": {"caudal": -24.0, "crest": 24.3, "side": 22.0},
    "forge_floatzel_actions": {"fang": 27.0, "nang": 27.0, "thi": 22.0, "tlo": 7.0},
    "forge_froslass_mega_actions": {"hair": 22.0, "hem": 4.0, "horn": 22.0, "jaw": 1.0, "sleeve": 22.0, "trail": 6.0},
    "forge_gabite_actions": {"farm": -22.0, "fin": 22.0, "jaw": 3.0, "narm": 7.0, "tail": -5.0},
    "forge_gallade_mega_actions": {"blade": 27.0, "cape": 6.0, "jaw": 2.0, "leg": 22.0},
    "forge_garchomp_actions": {"fin_crest": 7.0, "fin_dorsal": 22.0, "jaw": 32.0, "leg_front": 28.0, "leg_rear": 26.0, "scythe_far": 22.0, "scythe_near": -22.0, "tail": 22.0},
    "forge_garchomp_mega_actions": {"arm": -22.0, "fin": 22.0, "leg": 22.0, "snout": 22.0},
    "forge_garchomp_mega_z_actions": {"bank": 6.0, "crest": 24.3, "jaw": 3.0, "tail": 22.0, "wing": 22.0},
    "forge_gible_actions": {"arm": -22.0, "fin": 22.0, "jaw": 3.0, "tail": 5.0},
    "forge_glaceon_actions": {"ear_flat": 22.0, "ear_up": 24.3, "lift": 22.0, "ribbon_f": 30.0, "ribbon_l": 38.0, "tail": -24.0},
    "forge_glameow_actions": {"ear": 22.0, "flare": 2.0, "tail": 6.0},
    "forge_grotle_actions": {"crest_l": 6.0, "crest_r": 5.0, "leg_f": 26.0, "leg_m": 22.0, "leg_r": 22.0},
    "forge_happiny_actions": {"curl": 27.0},
    "forge_heatran_mega_actions": {"jaw": 54.0, "ll_sw": 22.0, "lr_sw": 22.0},
    "forge_hippopotas_actions": {"jaw": 3.0},
    "forge_hippowdon_actions": {"dorsal": 22.0, "jaw": 3.0, "tail": -5.0},
    "forge_honchkrow_actions": {"crest": 22.0, "jaw": 3.0, "plume": -22.0, "tal_l": 32.0, "tal_r": -28.0},
    "forge_infernape_actions": {"arm_low": -22.0, "arm_up": -22.0, "flame": 24.3, "flare": 4.0, "tail": 22.0},
    "forge_kricketot_actions": {"ant_l": 22.0, "ant_r": -22.0, "jaw": 2.0},
    "forge_kricketune_actions": {"ant_l": 22.0, "ant_r": -22.0, "blade_l": 24.3, "blade_r": -22.0, "mouth": 2.0},
    "forge_leafeon_actions": {"ear_l": 4.0, "tail": 22.0},
    "forge_lickilicky_actions": {"curl": 5.0, "mouth": 1.0},
    "forge_lopunny_actions": {"ear": -22.0},
    "forge_lopunny_mega_actions": {"arm": -22.0, "ears": 22.0, "leg": 22.0, "side": 22.0},
    "forge_lucario_mega_actions": {"arm_l": 22.0, "arm_r": 5.0, "crest": 22.0, "leg": 22.0, "mouth": 3.0, "snout": 22.0},
    "forge_lucario_mega_z_actions": {"crest": 22.0, "far": -22.0, "fore": 22.0, "jaw": 3.0, "near": 22.0, "rear": 22.0},
    "forge_lumineon_actions": {"crest": 24.3, "flare": 1.0, "pec": 22.0, "tail": -22.0, "wing": -24.0},
    "forge_luxio_actions": {"jaw": 2.0, "tail": -22.0},
    "forge_luxray_actions": {"fore": 22.0, "hind": -24.0, "jaw": 2.0, "mid": 24.3, "tail": 22.0},
    "forge_magmortar_actions": {"arm_l": 70.0, "arm_r": -42.0, "flame_l": -22.0, "flame_r": 22.0, "flare": 4.0, "jaw": 3.0, "tail": 22.0},
    "forge_magnezone_actions": {"ant": 5.0, "mag": 7.0},
    "forge_mantyke_actions": {"left_ant": 24.3, "left_wing": 30.0, "mouth": 1.0, "right_ant": -24.3, "right_wing": -30.0, "tail": -22.0},
    "forge_mismagius_actions": {"sway": 7.0},
    "forge_monferno_actions": {"flame": 26.0, "flare": 2.0, "left_arm": 24.3, "right_arm": -24.3, "tail": 22.0},
    "forge_mothim_actions": {"ant": 22.0, "wing": 22.0},
    "forge_munchlax_actions": {"jaw": 2.0},
    "forge_pachirisu_actions": {"tail": 22.0},
    "forge_piplup_actions": {"flip_l": 22.0, "flip_r": -24.3, "jaw": 3.0},
    "forge_prinplup_actions": {"crest_l": -22.0, "crest_r": 22.0, "jaw": 2.0},
    "forge_purugly_actions": {"lpaw": -27.0, "rpaw": -27.0, "tail": 22.0},
    "forge_rampardos_actions": {"arm": 22.0, "jaw": 34.0, "tail": 22.0},
    "forge_rhyperior_actions": {"drill": 27.0, "tail": 7.0},
    "forge_riolu_actions": {"arm_l": 22.0, "arm_r": 22.0, "ear_l": 22.0, "ear_r": 22.0, "flap": 22.0, "jaw": 2.0, "tail": 6.0},
    "forge_roserade_actions": {"flare": 2.0, "rose_blue": 22.0, "rose_pink": 22.0},
    "forge_shellos_actions": {"open": 22.0},
    "forge_shieldon_actions": {"tail": 22.0},
    "forge_shinx_actions": {"jaw": 2.0, "tail": 22.0},
    "forge_skorupi_actions": {"left": -22.0, "right": 22.0, "tail": -22.0},
    "forge_skuntank_actions": {"tail": -38.0},
    "forge_snover_actions": {"branch_l": 22.0, "branch_r": 22.0},
    "forge_spiritomb_actions": {"sway": 8.0},
    "forge_staraptor_actions": {"crest": 22.0, "jaw": 3.0, "tal_l": 7.0, "tal_r": -6.0, "wing": -22.0},
    "forge_staraptor_mega_actions": {"crest": 22.0, "jaw": 3.0, "tail": 22.0, "talon": 4.0, "wing": 22.0},
    "forge_staravia_actions": {"jaw": 3.0},
    "forge_starly_actions": {"crest": 22.0, "wing": 22.0},
    "forge_stunky_actions": {"tail": 22.0},
    "forge_tangrowth_actions": {"flare": 2.0},
    "forge_togekiss_actions": {"jaw": 2.0},
    "forge_torterra_actions": {"jaw": 3.0},
    "forge_toxicroak_actions": {"arm_l": 22.0, "arm_r": 22.0, "crest": 22.0, "jaw": 2.0, "throat": 4.0},
    "forge_turtwig_actions": {"jaw": 2.0, "leaf": 22.0},
    "forge_vespiquen_actions": {"arm_l": 22.0, "arm_r": 22.0, "wing": 22.0},
    "forge_weavile_actions": {"arm_l": -28.0, "arm_r": -24.0, "ear_l": 6.0, "ear_r": -5.0, "tail": -22.0},
    "forge_wormadam_plant_actions": {"antenna": 22.0, "arm_l": 22.0, "arm_r": 22.0, "flare": 2.0},
    "forge_yanmega_actions": {"wing": 22.0},
}

def finish(p, action, f, script_file):
    """Drop the in-plane spin and extend the species strike."""
    if action != "attack":
        return p
    if "spin" in p:
        p["spin"] = 0
    if "rigid" in p and 5 <= f <= 10:
        try:
            if abs(float(p["rigid"])) > 18:
                p["rigid"] = 0.0
        except (TypeError, ValueError):
            pass
    if "spin_dy" in p and 5 <= f <= 10:
        p["spin_dy"] = 0
    stem = Path(script_file).stem
    table = PEAKS.get(stem)
    if not table or not (0 <= f < len(ENV)):
        return p
    env = ENV[f]
    if env <= 0:
        return p
    for key, peak in table.items():
        if key not in p:
            continue
        cur = p[key]
        if isinstance(cur, (tuple, list, str)):
            continue
        try:
            cur_f = float(cur)
        except (TypeError, ValueError):
            continue
        target = float(peak) * env
        if abs(target) <= abs(cur_f) + 0.05:
            continue
        if isinstance(cur, int) and abs(float(peak)) <= 8:
            p[key] = int(round(target))
        else:
            p[key] = target
    return p
