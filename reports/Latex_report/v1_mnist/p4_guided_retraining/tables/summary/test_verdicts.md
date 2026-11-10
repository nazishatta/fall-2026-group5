# Phase 4 test verdicts

Copied unchanged from the EC2 run output (test_results.json of each run).

```text
PHASE 4 TEST VERDICTS
Source: test_results.json of each run (test_comparisons.csv holds the same values rounded to 6 decimals).
Differences: test error of first arm minus second arm, mean over seeds 1-5.
Intervals: 95% paired bootstrap over test images, 2000 resamples, seed 2026.
Macro-F1 guard: 0.0015657542982220032   Stamp SHA-256: 5de303af876671d9cbe6299a9954691a4b1d885c3e12329ac9e6fb920f78fa9d

====================================================================================================
1. SUCCESS CRITERIA (RETRAINING_PROTOCOL.md, section 5)
====================================================================================================

C1  RULE: (curriculum helps): `curriculum_som` has a lower overall test error than `standard`, the interval excludes 0, and the macro-F1 guard holds.
    experiment          : curriculum_comparison_som_15x15_nb11_ep250_s42_v2
    comparison / subset : curriculum_som minus standard / overall  (n=10500)
    mean difference     : +0.004476
    95% bootstrap CI    : [+0.003276, +0.005696]   excludes_zero=True
    macro-F1 difference : -0.004438   macro_f1_guard_holds=False
    meets_overall_criterion = False
    VERDICT             : NOT MET

C2  RULE: (SOM guidance matters): `curriculum_som` has a lower hotspot test error than `curriculum_random`, the interval excludes 0, and the guard holds.
    experiment          : curriculum_comparison_som_15x15_nb11_ep250_s42_v2
    comparison / subset : curriculum_som minus curriculum_random / hotspot  (n=146)
    mean difference     : +0.091781
    95% bootstrap CI    : [+0.054795, +0.130171]   excludes_zero=True
    macro-F1 difference : -0.004023   macro_f1_guard_holds=False
    meets_criterion = False
    VERDICT             : NOT MET

A1  RULE: (primary, SOM guidance matters): `augmented_som` has a lower hotspot test error than `augmented_random`, the interval excludes 0, and the guard holds.
    experiment          : augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2
    comparison / subset : augmented_som minus augmented_random / hotspot  (n=146)
    mean difference     : -0.024658
    95% bootstrap CI    : [-0.045205, -0.006849]   excludes_zero=True
    macro-F1 difference : +0.000311   macro_f1_guard_holds=True
    meets_criterion = True
    VERDICT             : MET

A2  RULE: (augmentation helps the hotspots): `augmented_som` has a lower hotspot test error than `standard`, the interval excludes 0, and the guard holds.
    experiment          : augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2
    comparison / subset : augmented_som minus standard / hotspot  (n=146)
    mean difference     : -0.035616
    95% bootstrap CI    : [-0.057534, -0.013699]   excludes_zero=True
    macro-F1 difference : +0.001037   macro_f1_guard_holds=True
    meets_criterion = True
    VERDICT             : MET

A3  RULE: (no damage elsewhere): the upper end of the interval of (`augmented_som` minus `standard`) on the non-hotspot test error is at most 0.002.
    experiment          : augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2
    comparison / subset : augmented_som minus standard / non_hotspot  (n=10354)
    mean difference     : -0.000522
    95% bootstrap CI    : [-0.001314, +0.000232]   excludes_zero=False
    macro-F1 difference : +0.001037   macro_f1_guard_holds=True
    no_damage_elsewhere = True
    VERDICT             : MET

====================================================================================================
2. FULL COMPARISON TABLES
====================================================================================================
Columns: guard = macro_f1_guard_holds, hot = meets_criterion, ovr = meets_overall_criterion, dmg = no_damage_elsewhere

--- curriculum_comparison_som_15x15_nb11_ep250_s42_v2 ---
comparison                                       overall diff [95% CI]             hotspot diff [95% CI]             non-hotspot diff [95% CI]         macroF1 diff guard  hot    ovr    dmg
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
curriculum_msp minus standard                    +0.004190 [+0.002990, +0.005372]  +0.112329 [+0.071233, +0.154795]  +0.002666 [+0.001603, +0.003825]  -0.004201    False  False  False  False
curriculum_random minus standard                 +0.000419 [-0.000248, +0.001086]  +0.008219 [-0.015068, +0.031507]  +0.000309 [-0.000290, +0.000908]  -0.000415    True   False  False  True
curriculum_som minus curriculum_msp              +0.000286 [-0.000895, +0.001486]  -0.012329 [-0.058904, +0.036986]  +0.000464 [-0.000579, +0.001468]  -0.000237    True   False  False  True
curriculum_som minus curriculum_random           +0.004057 [+0.002933, +0.005219]  +0.091781 [+0.054795, +0.130171]  +0.002820 [+0.001796, +0.003941]  -0.004023    False  False  False  False
curriculum_som minus standard                    +0.004476 [+0.003276, +0.005696]  +0.100000 [+0.063014, +0.136986]  +0.003129 [+0.002028, +0.004211]  -0.004438    False  False  False  False

--- augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2 ---
comparison                                       overall diff [95% CI]             hotspot diff [95% CI]             non-hotspot diff [95% CI]         macroF1 diff guard  hot    ovr    dmg
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
augmented_hotspot minus augmented_random_hotspot -0.000190 [-0.000876, +0.000439]  -0.021918 [-0.045205, +0.001370]  +0.000116 [-0.000425, +0.000676]  +0.000187    True   False  False  True
augmented_hotspot minus standard                 -0.000286 [-0.000971, +0.000419]  -0.019178 [-0.043836, +0.006849]  -0.000019 [-0.000618, +0.000541]  +0.000285    True   False  False  True
augmented_msp minus standard                     -0.001390 [-0.002343, -0.000419]  -0.028767 [-0.061644, +0.002774]  -0.001004 [-0.001816, -0.000212]  +0.001402    True   False  True   True
augmented_random minus standard                  -0.000705 [-0.001333, -0.000057]  -0.010959 [-0.034247, +0.013733]  -0.000560 [-0.001140, +0.000000]  +0.000725    True   False  True   True
augmented_random_hotspot minus standard          -0.000095 [-0.000686, +0.000514]  +0.002740 [-0.015068, +0.021918]  -0.000135 [-0.000657, +0.000387]  +0.000098    True   False  False  True
augmented_random_paced minus augmented_random    +0.000743 [+0.000114, +0.001390]  +0.012329 [-0.008219, +0.032877]  +0.000579 [-0.000039, +0.001140]  -0.000776    True   False  False  True
augmented_random_paced minus standard            +0.000038 [-0.000552, +0.000629]  +0.001370 [-0.017842, +0.020548]  +0.000019 [-0.000522, +0.000579]  -0.000051    True   False  False  True
augmented_som minus augmented_msp                +0.000381 [-0.000419, +0.001124]  -0.006849 [-0.028767, +0.012329]  +0.000483 [-0.000212, +0.001217]  -0.000365    True   False  False  True
augmented_som minus augmented_random             -0.000305 [-0.001067, +0.000419]  -0.024658 [-0.045205, -0.006849]  +0.000039 [-0.000676, +0.000753]  +0.000311    True   True   False  True
augmented_som minus standard                     -0.001010 [-0.001810, -0.000171]  -0.035616 [-0.057534, -0.013699]  -0.000522 [-0.001314, +0.000232]  +0.001037    True   True   True   True
augmented_som_paced minus augmented_random_paced -0.000133 [-0.000781, +0.000552]  -0.002740 [-0.024658, +0.019178]  -0.000097 [-0.000637, +0.000483]  +0.000164    True   False  False  True
augmented_som_paced minus augmented_som          +0.000914 [+0.000152, +0.001657]  +0.034247 [+0.016438, +0.053425]  +0.000444 [-0.000252, +0.001159]  -0.000923    True   False  False  True
augmented_som_paced minus standard               -0.000095 [-0.000819, +0.000590]  -0.001370 [-0.027397, +0.023288]  -0.000077 [-0.000734, +0.000560]  +0.000114    True   False  False  True

====================================================================================================
3. TEST IMAGES IN HOTSPOT NEURONS
====================================================================================================
curriculum_comparison_som_15x15_nb11_ep250_s42_v2       : 146 of 10500 test images
augmentation_comparison_som_15x15_nb11_ep250_s42_v2_r2  : 146 of 10500 test images
```
