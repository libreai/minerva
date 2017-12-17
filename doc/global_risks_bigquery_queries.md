
#### Risk_001_asset_bubbles_in_a_major_economy
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%FINANCIAL_CRISIS%' or
  V2Themes like '%ECON_BUBBLE%' or
  V2Themes like '%EPU_CATS_SOVEREIGN_DEBT_CURRENCY_CRISES%' or
  V2Themes like '%WB_1142_FINANCIAL_SECTOR_INSTABILITY%' or
  V2Themes like '%WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT%'
```

#### Risk_002_deflation_in_a_major_economy
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%DEFLATION%' or
  V2Themes like '%WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT%'
```

#### Risk_003_failure_of_a_major_financial_mechanism_or_institution
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_1068_MARKET_FAILURES_VERSUS_GOVERNMENT_FAILURES%' or
  V2Themes like '%FINANCIAL_CRISIS%' or
  V2Themes like '%WB_1142_FINANCIAL_SECTOR_INSTABILITY"%'
```

#### Risk_004_failure_shortfall_of_critical_infrastructure
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
V2Themes like '%MANMADE_DISASTER_TRANSPORTATION_DISASTER%' or
V2Themes like '%INTERNET_BLACKOUT%' or
V2Themes like '%MANMADE_DISASTER_POWER_BLACKOUT%' or
V2Themes like '%POWER_OUTAGE%' or
V2Themes like '%MANMADE_DISASTER_POWER_OUTAGES%' or
V2Themes like '%PHONE_OUTAGE%' or
V2Themes like '%WB_____WATER_SUPPLY%' or
V2Themes like '%WB______WATER_ALLOCATION_AND_WATER_SUPPLY%'
```

#### Risk_005_fiscal_crises_in_key_economies
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_1104_MACROECONOMIC_VULNERABILITY_AND_DEBT%'
```

#### Risk_006_high_structural_unemployment_or_underemployment
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%UNEMPLOYMENT%' or
  V2Themes like '%WB_2747_UNEMPLOYMENT%' or
  V2Themes like '%WB_1649_UNEMPLOYMENT_BENEFITS%' or
  V2Themes like '%WB_2678_CYCLICAL_UNEMPLOYMENT%' or
  V2Themes like '%WB_2809_UNEMPLOYMENT_INSURANCE_REFORMS%' or
  V2Themes like '%WB______UNEMPLOYMENT%' or
  V2Themes like '%WB_2670_JOBS%' or
  V2Themes like '%WB_2689_JOBS_DIAGNOSTICS%' or
  V2Themes like '%WB_2131_EMPLOYABILITY_SKILLS_AND_JOBS%' or
  V2Themes like '%WB_2745_JOB_QUALITY_AND_LABOR_MARKET_PERFORMANCE%' or
  V2Themes like '%WB_2769_JOBS_STRATEGIES%' or
  V2Themes like '%WB_855_LABOR_MARKETS%' or
  V2Themes like '%WB_2836_MIGRATION_POLICIES_AND_JOBS%' or
  V2Themes like '%UNGP_JOB_OPPORTUNITIES_EMPLOYMENT%' or
  V2Themes like '%WB_2823_ON_THE_JOB_TRAINING%' or
  V2Themes like '%WB_701_JOBS_AND_POVERTY%' or
  V2Themes like '%WB_1170_JOB_CREATION_AND_JOB_OPPORTUNITIES%' or
  V2Themes like '%UNGP_JOB_OPPORTUNITIES_WORKING_CONDITIONS%' or
  V2Themes like '%WB_2884_INCLUSIVE_JOBS%' or
  V2Themes like '%WB_2420_ICT_FOR_JOBS%' or
  V2Themes like '%WB_2889_MINORITIES_AND_DISENFRANCHISED_GROUPS_AND_JOBS%' or
  V2Themes like '%WB_2671_JOBS_AND_DEVELOPMENT%' or
  V2Themes like '%WB_2683_CHANGING_NATURE_OF_JOBS%' or
  V2Themes like '%WB_2773_FISCAL_POLICY_AND_JOBS%' or
  V2Themes like '%WB_2674_GREEN_JOBS%' or
  V2Themes like '%WB_2673_JOBS_AND_CLIMATE_CHANGE%' or
  V2Themes like '%WB_2724_PUBLIC_SECTOR_JOBS%' or
  V2Themes like '%WB_1580_ON_THE_JOB_TRAINING%' or
  V2Themes like '%TAX_FNCACT_JOBHOLDER%' or
  V2Themes like '%WB_1660_JOB_SHARING%' or
  V2Themes like '%WB_2862_JOBS_IN_TOURISM%' or
  V2Themes like '%WB_2861_JOBS_IN_TRADE_AND_COMPETITIVENESS%' or
  V2Themes like '%WB_475_JOBS_AND_GROWTH%' or
  V2Themes like '%WB_2679_JOBLESS_GROWTH%' or
  V2Themes like '%WB_2849_JOBS_IN_AGRICULTURE%' or
  V2Themes like '%WB_2706_JOBS_IN_FRAGILE_STATES%' or
  V2Themes like '%WB_2783_JOB_TAX_CREDITS%' or
  V2Themes like '%WB_2850_JOBS_IN_EDUCATION%' or
  V2Themes like '%WB_1659_JOB_ROTATION%' or
  V2Themes like '%TAX_FNCACT_JOBHOLDERS%' or
  V2Themes like '%WB______JOBS%'
```

#### Risk_007_illicit_trade
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
V2Themes like '%CRIME_ILLEGAL_DRUGS%' or
V2Themes like '%ORGANIZED_CRIME%' or
V2Themes like '%CRIME_CARTELS%' or
V2Themes like '%WB_2075_TAX_CRIME%' or
V2Themes like '%WB_2455_CRIME_NETWORKS%' or
V2Themes like '%WB______ORGANIZED_CRIME%' or
V2Themes like '%WB_698_TRADE%' or
V2Themes like '%DRUG_TRADE%' or
V2Themes like '%WB_775_TRADE_POLICY_AND_INTEGRATION%' or
V2Themes like '%WB_2601_TRADE_LINKAGES_SPILLOVERS_AND_CONNECTIVITY%' or
V2Themes like '%ECON_TRADE_DISPUTE%' or
V2Themes like '%WB_1136_TRADE_BALANCE%' or
V2Themes like '%WB_1192_IMPACT_OF_TRADE%' or
V2Themes like '%HUMAN_TRAFFICKING%' or
V2Themes like '%WB_2458_HUMAN_TRAFFICKING%' or
V2Themes like '%WB_2461_TRAFFICKING_NATURAL_RESOURCES%' or
V2Themes like '%WB_2073_ILLICIT_FINANCIAL_FLOWS%' or
V2Themes like '%WB_2459_ILLICIT_FINANCIAL_FLOWS%' or
V2Themes like '%WB______ILLICIT_FINANCIAL_FLOWS%'
```

#### Risk_008_severe_energy_price_shock
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_507_ENERGY_AND_EXTRACTIVES%' or
  V2Themes like '%WB_1627_ENERGY_SUBSIDIES%' or
  V2Themes like '%WB_1767_ENERGY_FINANCE%' or
  V2Themes like '%WB_521_ENERGY_ACCESS%' or
  V2Themes like '% WB_1697_ENERGY_EFFICIENCY_FINANCE%' or
  V2Themes like '%WB_____ENERGY_AND_EXTRACTIVES%' or
  V2Themes like '%WB_534_EFFICIENT_ENERGY_SUPPLY%' or
  V2Themes like '%ECON_ELECTRICALPRICE%'
```

#### Risk_009_unmanageable_inflation
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%TAX_ECON_PRICE%' or
  V2Themes like '%ECON_HOUSING_PRICES%' or
  V2Themes like '%ECON_OILPRICE%' or
  V2Themes like '%FUELPRICES%' or
  V2Themes like '%ECON_GASOLINEPRICE%' or
  V2Themes like '% ECON_GOLDPRICE%' or
  V2Themes like '%ECON_ELECTRICALPRICE%' or
  V2Themes like '%ECON_NATGASPRICE%' or
  V2Themes like '%ECON_DIESELPRICE%' or
  V2Themes like '%ECON_PRICECONTROL%' or
  V2Themes like '%WB_2107_PRICE_CONTROLS%' or
  V2Themes like '%ECON_PRICEGOUGE%' or
  V2Themes like '%ECON_HEATINGOILPRICE%' or
  V2Themes like '%ECON_PRICEMANIPULATION%' or
  V2Themes like '%WB_200_FOOD_PRICE_ANALYSIS%' or
  V2Themes like '%ECON_PROPANEPRICE%' or
  V2Themes like '%WB______PRICE_SUBSIDIES%' or
  V2Themes like '%WB_1164_COMMODITY_PRICES_SHOCKS%' or
  V2Themes like '%WB______PRICE_CONTROLS%'
```

#### Risk_010_extreme_weather_events
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%NATURAL_DISASTER_%'
```

#### Risk_011_failure_of_climate_change_mitigation_and_adaptation
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%ENV_CLIMATECHANGE%' or
  V2Themes like '%WB_566_ENVIRONMENT_AND_NATURAL_RESOURCES%' or
  V2Themes like '%WB_1786_ENVIRONMENTAL_SUSTAINABILITY%' or
  V2Themes like '%WB_598_ENVIRONMENTAL_MANAGEMENT%' or
  V2Themes like '%MOVEMENT_ENVIRONMENTAL%' or
  V2Themes like '%WB_901_ENVIRONMENTAL_SAFEGUARDS%' or
  V2Themes like '%WB_2197_ENVIRONMENTAL_ENGINEERING%' or
  V2Themes like '%WB_1792_ENVIRONMENTAL_HEALTH%' or
  V2Themes like '%WB_849_ENVIRONMENTAL_LAWS_AND_REGULATIONS%' or
  V2Themes like '%SELF_IDENTIFIED_ENVIRON_DISASTER%' or
  V2Themes like '%WB_1785_ENVIRONMENTAL_POLICIES_AND_INSTITUTIONS%' or
  V2Themes like '%WB_1388_ENVIRONMENTAL_AND_SOCIAL_ASSESSMENTS%' or
  V2Themes like '%MANMADE_DISASTER_ENVIRONMENTAL_DISASTER%' or
  V2Themes like '%WB_1831_ENVIRONMENTAL_CRIME_AND_LAW_ENFORCEMENT%' or
  V2Themes like '%WB_1717_URBAN_POLLUTION_AND_ENVIRONMENTAL_HEALTH%' or
  V2Themes like '%WB_2915_ENVIRONMENTAL_CRIME%' or
  V2Themes like '%TAX_FNCACT_ENVIRONMENTAL_SCIENTIST%' or
  V2Themes like '%ECON_DEVELOPMENTORGS_UNITED_NATIONS_ENVIRONMENT_PROGRAMME%' or
  V2Themes like '%WB_2916_ENVIRONMENTAL_LAW_ENFORCEMENT%' or
  V2Themes like '%ECON_DEVELOPMENTORGS_UNITED_NATIONS_ENVIRONMENT_PROGRAM%' or
  V2Themes like '%WB_1782_ENVIRONMENTAL_AGREEMENTS_AND_CONVENTIONS%' or
  V2Themes like '%WB_1783_ENVIRONMENTAL_GOVERNANCE%' or
  V2Themes like '%WB_2307_ENVIRONMENTAL_MANAGEMENT_AND_MITIGATION_PLANS%' or
  V2Themes like '%WB_2306_ENVIRONMENTAL_IMPACT_ASSESSEMENT%' or
  V2Themes like '%WB_1376_ENVIRONMENTAL_OFFSETS%' or
  V2Themes like '%WB______BUSINESS_ENVIRONMENT%' or
  V2Themes like '%WB_____ENVIRONMENT_AND_NATURAL_RESOURCES%' or
  V2Themes like '%WB_2205_ENVIRONMENTAL_AND_SOCIAL_CATEGORIZATION%'
```

#### Risk_012_major_biodiversity_loss_and_ecosystem_collapse
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_2084_BIODIVERSITY%' or
  V2Themes like '%ENV_SPECIESENDANGERED%' or
  V2Themes like '%ENV_SPECIESEXTINCT%' or
  V2Themes like '%WB_435_AGRICULTURE_AND_FOOD_SECURITY%' or
  V2Themes like '%WB______BIODIVERSITY%'
```

#### Risk_013_man_made_environmental_damage_and_disasters
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%MANMADE_DISASTER_OIL_SPILL%' or
  V2Themes like '%MANMADE_DISASTER_FIRE_TRUCKS%' or
  V2Themes like '%MANMADE_DISASTER_FIRE_TRUCK%' or
  V2Themes like '%MANMADE_*%'
```

#### Risk_014_failure_of_national_governance
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_2020_BRIBERY_FRAUD_AND_COLLUSION%' or
  V2Themes like '%ELECTION_FRAUD%' or
  V2Themes like '%TAX_FNCACT_FRAUD_EXAMINER%' or
  V2Themes like '%WB______BRIBERY_FRAUD_AND_COLLUSION%' or
  V2Themes like '%WB_1068_MARKET_FAILURES_VERSUS_GOVERNMENT_FAILURES%' or
  V2Themes like '%GOV_STATE_FAILURE%' or
  V2Themes like '%WB_1069_RESOURCE_MISALLOCATIONS_AND_POLICY_FAILURES%' or
  V2Themes like '%SLFID_RULE_OF_LAW%' or
  V2Themes like '%WB_2462_POLITICAL_VIOLENCE_AND_WAR%' or
  V2Themes like '% POLITICAL_TURMOIL%' or
  V2Themes like '%WB_739_POLITICAL_VIOLENCE_AND_CIVIL_WAR%' or
  V2Themes like '%WB_926_POLITICAL_PARTICIPATION%' or
  V2Themes like '%WB______POLITICAL_VIOLENCE_AND_WAR%' or
  V2Themes like '%WB_____POLITICAL_VIOLENCE_AND_CIVIL_WAR%' or
  V2Themes like '%TAX_TERROR_GROUP_LIKUD_POLITICAL_PARTIES%' or
  V2Themes like '%WB_1095_POLITICAL_AND_INSTITUTIONAL_SUSTAINABILITY%' or
  V2Themes like '%WB_832_ANTI_CORRUPTION%' or
  V2Themes like '%WB_2024_ANTI_CORRUPTION_AUTHORITIES%' or
  V2Themes like '%CORRUPTION%' or
  V2Themes like '%WB_2019_ANTI_CORRUPTION_LEGISLATION%' or
  V2Themes like '%WB_2595_ANTI_CORRUPTION_IN_CUSTOMS_ADMINISTRATION%' or
  V2Themes like '%WB_313_INTEGRITY_AND_ANTI_CORRUPTION%' or
  V2Themes like '%WB_____ANTI_CORRUPTION%' or
  V2Themes like '%WB______ANTI_CORRUPTION_AUTHORITIES%' or
  V2Themes like '%WB_2028_ANTI_CORRUPTION_PREVENTION%' or
  V2Themes like '%WB______ANTI_CORRUPTION_LEGISLATION%' or
  V2Themes like '%TAX_TERROR_GROUP_NATIONAL_ANTICORRUPTION_FRONT%' or
  V2Themes like '%WB______ANTI_CORRUPTION_IN_CUSTOMS_ADMINISTRATION%' or
  V2Themes like '%TAX_FNCACT_COUP_PLOTTERS%' or
  V2Themes like '%TAX_FNCACT_COUP_PLOTTER%' or
  V2Themes like '%WB_737_LOCAL_GOVERNANCE%'
```

#### Risk_015_failure_of_regional_or_global_governance
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_831_GOVERNANCE%' or
  V2Themes like '%WB_813_URBAN_GOVERNANCE_AND_CITY_SYSTEMS%' or
  V2Themes like '%WB_1385_GOVERNANCE_AND_STEWARDSHIP%' or
  V2Themes like '%WB_1922_GOVERNANCE_STRUCTURES%' or
  V2Themes like '%MANMADE_DISASTER_ENVIRONMENTAL_DISASTER%'
```

#### Risk_016_interstate_conflict_with_regional_consequences
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%"ECON_TRADE_DISPUTE%' or
  V2Themes like '%WB_1883_CONTRACT_DISPUTES%' or
  V2Themes like '%WB_1141_ECONOMIC_SHOCKS_AND_TRADE%' or
  V2Themes like '%ARMEDCONFLICT%' or
  V2Themes like '%WB_2969_SOCIAL_CONFLICT%' or
  V2Themes like '%WB______FRAGILITY_CONFLICT_AND_VIOLENCE%' or
  V2Themes like '%WB______CONFLICT_AND_VIOLENCE"%'
```

#### Risk_017_large_scale_terrorist_attacks
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%"TAX_TERROR_GROUP_ETA_TERRORISTS%' or
  V2Themes like '%TAX_TERROR_GROUP_ETA_TERRORIST%' or
  V2Themes like '%TAX_TERROR_GROUP_AUTONOMOUS_COMBAT_TERRORIST_ORGANIZATION%' or
  V2Themes like '%TAX_TERROR_GROUP_EVAN_MECHAM_ECO_TERRORIST_INTERNATIONAL%' or
  V2Themes like '%TAX_TERROR_GROUP_MAKHACHKALA_TERRORISTS%' or
  V2Themes like '%TAX_TERROR_GROUP_ISLAMIC_STATE%' or
  V2Themes like '%TAX_TERROR_GROUP_ISLAMIC_STATE_OF_IRAQ"%'
```

#### Risk_018_state_collapse_or_crisis
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%"GOV_STATE_FAILURE%' or
  V2Themes like '%WB_2969_SOCIAL_CONFLICT%'
```

#### Risk_019_weapons_of_mass_destruction
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB______WEAPONS_OF_MASS_DESTRUCTION%'
```

#### Risk_020_failure_of_urban_planning
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_699_URBAN_DEVELOPMENT%' or
  V2Themes like '%WB_813_URBAN_GOVERNANCE_AND_CITY_SYSTEMS%' or
  V2Themes like '%WB_797_NATIONAL_URBAN_POLICIES%' or
  V2Themes like '%WB_788_URBAN_TRANSPORT%' or
  V2Themes like '%WB_816_STRATEGIC_URBAN_PLANNING%' or
  V2Themes like '%WB_1835_URBAN_REGENERATION%' or
  V2Themes like '%URBAN_SPRAWL%' or
  V2Themes like '%WB_804_URBAN_POLLUTION%' or
  V2Themes like '%WB_1717_URBAN_POLLUTION_AND_ENVIRONMENTAL_HEALTH%' or
  V2Themes like '%WB_785_URBAN_ROADS%' or
  V2Themes like '%WB_710_URBAN_POVERTY%' or
  V2Themes like '%WB_1781_URBAN_ECOSYSTEMS%'
```

#### Risk_021_food_crises
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_435_AGRICULTURE_AND_FOOD_SECURITY%'
```

#### Risk_022_large_scale_involuntary_migration
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%EPU_CATS_MIGRATION_FEAR_FEAR%' or
  V2Themes like '%EPU_CATS_MIGRATION_FEAR_MIGRATION%' or
  V2Themes like '%IMMIGRATION%' or
  V2Themes like '%WB_2837_IMMIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_XENOPHOBIC%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_XENOPHOBIA%' or
  V2Themes like '%WB_2844_EMIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ANTIIMMIGRATION%' or
  V2Themes like '%ECON_DEVELOPMENTORGS_INTERNATIONAL_ORGANIZATION_FOR_MIGRATION%' or
  V2Themes like '%TAX_AIDGROUPS_INTERNATIONAL_ORGANIZATION_FOR_MIGRATION%' or
  V2Themes like '%EPU_MIGRATION_FEAR_FEAR%' or
  V2Themes like '%SOC_MASSMIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ULTRANATIONALIST%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_AGAINST_IMMIGRANTS%' or
  V2Themes like '%EPU_MIGRATION_FEAR_MIGRATION%' or
  V2Themes like '%WB_2204_IN_MIGRATION%' or
  V2Themes like '%TAX_AIDGROUPS_INTERNATIONAL_ORGANISATION_FOR_MIGRATION%' or
  V2Themes like '%HUMAN_RIGHTS_ABUSES_FORCED_MIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ANTI_IMMIGRANT%' or
  V2Themes like '%HUMAN_RIGHTS_ABUSES_FORCED_MIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ANTI_IMMIGRANT%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ANTI_IMMIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ANTI_IMMIGRANTS%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ULTRA_NATIONALIST%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_XENOPHOBE%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_XENOPHOBES%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ULTRANATIONALISTS%' or
  V2Themes like '%HUMAN_RIGHTS_ABUSES_FORCED_MIGRATIONS%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ATTACKS_ON_IMMIGRANTS%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_OPPOSED_TO_IMMIGRATION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ATTACKS_AGAINST_IMMIGRANTS%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ULTRA_NATIONALISTS%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_ULTRA_NATIONALISTS%' or
  V2Themes like '%WB_2838_RURAL_TO_URBAN_MIGRATION%' or
  V2Themes like '%TAX_AIDGROUPS_INTERNATIONAL_CATHOLIC_MIGRATION_COMMISSION%' or
  V2Themes like '%DISCRIMINATION_IMMIGRATION_OPPOSED_TO_IMMIGRANTS%' or
  V2Themes like '%WB_1333_HEALTH_WORKER_MIGRATION%' or
  V2Themes like '%WB______IMMIGRATION%' or
  V2Themes like '%WB______EMIGRATION%'
```

#### Risk_023_profound_social_instability
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%UNREST_BELLIGERENT%' or
  V2Themes like '%WB_2462_POLITICAL_VIOLENCE_AND_WAR%' or
  V2Themes like '%POLITICAL_TURMOIL%' or
  V2Themes like '%WB_739_POLITICAL_VIOLENCE_AND_CIVIL_WAR%'
```

#### Risk_024_rapid_and_massive_spread_of_infectious_diseases
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%"TAX_DISEASE%' or
  V2Themes like '%WB_1406_DISEASES%' or
  V2Themes like '%TAX_DISEASE_DISEASE%' or
  V2Themes like '%TAX_DISEASE_INFECTION%' or
  V2Themes like '%WB_1415_COMMUNICABLE_DISEASE%' or
  V2Themes like '%TAX_DISEASE_FLU%' or
  V2Themes like '%TAX_DISEASE_FEVER%' or
  V2Themes like '%TAX_DISEASE_OUTBREAK%' or
  V2Themes like '%TAX_DISEASE_BACTERIA%' or
  V2Themes like '%TAX_DISEASE_COUGH%' or
  V2Themes like '%HEALTH_SEXTRANSDISEASE%' or
  V2Themes like '%TAX_DISEASE_POISONING%' or
  V2Themes like '%TAX_DISEASE_EBOLA%' or
  V2Themes like '%TAX_DISEASE_ZIKA%' or
  V2Themes like '%TAX_DISEASE_INFECTIOUS%' or
  V2Themes like '%TAX_DISEASE_EPIDEMIC%' or
  V2Themes like '%TAX_DISEASE_SYNDROME%' or
  V2Themes like '%TAX_DISEASE_EMERGENCIES%' or
  V2Themes like '%TAX_DISEASE_SEIZURES%' or
  V2Themes like '%TAX_DISEASE_INFLUENZA%' or
  V2Themes like '%TAX_DISEASE_PLAGUE%' or
  V2Themes like '%TAX_DISEASE_PNEUMONIA%' or
  V2Themes like '%TAX_DISEASE_ANTICIPATION%' or
  V2Themes like '%TAX_DISEASE_ZIKA_VIRUS%' or
  V2Themes like '%TAX_DISEASE_MEASLES%' or
  V2Themes like '%TAX_DISEASE_DENGUE%' or
  V2Themes like '%TAX_DISEASE_DIARRHEA%' or
  V2Themes like '%TAX_DISEASE_CHOLERA%' or
  V2Themes like '%TAX_DISEASE_HEADACHE%' or
  V2Themes like '%SOC_CHRONICDISEASE%' or
  V2Themes like '%TAX_DISEASE_BACTERIAL%' or
  V2Themes like '%TAX_DISEASE_CONTAGIOUS%' or
  V2Themes like '%TAX_DISEASE_PATHOGENS%' or
  V2Themes like '%TAX_DISEASE_VOMITING%' or
  V2Themes like '%TAX_DISEASE_WEST_NILE_VIRUS%' or
  V2Themes like '%TAX_DISEASE_NAUSEA%' or
  V2Themes like '%TAX_DISEASE_GASTROINTESTINAL%' or
  V2Themes like '%TAX_DISEASE_DIZZINESS%' or
  V2Themes like '%TAX_DISEASE_SALMONELLA%' or
  V2Themes like '%TAX_DISEASE_MIGRAINE%' or
  V2Themes like '%TAX_DISEASE_SARS%' or
  V2Themes like '%TAX_DISEASE_CONVULSIONS%' or
  V2Themes like '%TAX_DISEASE_CHRONIC_DISEASE%' or
  V2Themes like '%TAX_CHRONICDISEASE_CHRONIC_DISEASE%' or
  V2Themes like '%TAX_DISEASE_DEHYDRATION%' or
  V2Themes like '%TAX_DISEASE_SHINGLES%' or
  V2Themes like '%TAX_DISEASE_TREMOR%' or
  V2Themes like '%TAX_DISEASE_CONSTIPATION%' or
  V2Themes like '%TAX_DISEASE_ANTHRAX%' or
  V2Themes like '%TAX_DISEASE_FOOD_POISONING%' or
  V2Themes like '%TAX_DISEASE_YELLOW_FEVER%' or
  V2Themes like '%TAX_DISEASE_HIV_AIDS%' or
  V2Themes like '%TAX_DISEASE_STARVATION%' or
  V2Themes like '%TAX_DISEASE_HEMORRHAGE%' or
  V2Themes like '%TAX_DISEASE_BIRD_FLU%' or
  V2Themes like '%TAX_DISEASE_DENGUE_FEVER%' or
  V2Themes like '%TAX_DISEASE_CHIKUNGUNYA%' or
  V2Themes like '%TAX_DISEASE_LYME_DISEASE%'
```

#### Risk_025_water_crises
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_137_WATER%' or
  V2Themes like '%CRISISLEX_C06_WATER_SANITATION%' or
  V2Themes like '%WB_1462_WATER_SANITATION_AND_HYGIENE%' or
  V2Themes like '%WATER_SECURITY%' or
  V2Themes like '%WB_1199_WATER_SUPPLY_AND_SANITATION%' or
  V2Themes like '%WB_139_SANITATION_AND_WASTEWATER%' or
  V2Themes like '%WB_140_AGRICULTURAL_WATER_MANAGEMENT%' or
  V2Themes like '%UNGP_CLEAN_WATER_SANITATION%' or
  V2Themes like '%WB_141_WATER_RESOURCES_MANAGEMENT%' or
  V2Themes like '%WB_1000_WATER_MANAGEMENT_STRUCTURES%' or
  V2Themes like '%WB_138_WATER_SUPPLY%' or
  V2Themes like '%WB_2008_WATER_TREATMENT%' or
  V2Themes like '%WB_1064_WATER_DEMAND_MANAGEMENT%' or
  V2Themes like '%WB_1798_WATER_POLLUTION%' or
  V2Themes like '%WB_1021_WATER_LAW%' or
  V2Themes like '%WB_144_URBAN_WATER%' or
  V2Themes like '%WB_1063_WATER_ALLOCATION_AND_WATER_SUPPLY%' or
  V2Themes like '%WB_143_RURAL_WATER%' or
  V2Themes like '%WB_1998_WATER_ECONOMICS%' or
  V2Themes like '%WB_1215_WATER_QUALITY_STANDARDS%' or
  V2Themes like '%WB_2971_WATER_PRICING%' or
  V2Themes like '%WB_149_WASTEWATER_TREATMENT_AND_DISPOSAL%' or
  V2Themes like '%WB_2981_DRINKING_WATER_QUALITY_STANDARDS%' or
  V2Themes like '%WB_2009_WATER_QUALITY_MONITORING%' or
  V2Themes like '%WB_150_WASTEWATER_REUSE%' or
  V2Themes like '%WB_1729_URBAN_WATER_FINANCIAL_SUSTAINABILITY%' or
  V2Themes like '%WB_1731_NON_REVENUE_WATER%' or
  V2Themes like '%WB_2007_WATER_SAFETY_PLANS%' or
  V2Themes like '%TAX_DISEASE_WATER_INTOXICATION%'
```

#### Risk_026_adverse_consequences_of_technological_advances
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%WB_376_INNOVATION_TECHNOLOGY_AND_ENTREPRENEURSHIP%' or
  V2Themes like '%SOC_TECHNOLOGYSECTOR%' or
  V2Themes like '%WB_1084_TECHNOLOGY_TRANSFER_AND_DIFFUSION%' or
  V2Themes like '%WB_1950_AGRICULTURE_TECHNOLOGY%' or
  V2Themes like '%WB_2377_TECHNOLOGY_ARCHITECTURE%' or
  V2Themes like '%WB_378_INNOVATION_AND_TECHNOLOGY_POLICY%' or
  V2Themes like '%WB_1952_MITIGATION_TECHNOLOGY%'
```

#### Risk_027_breakdown_of_critical_information_infrastructure_and_networks_and_cybrattacks
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%CYBER_ATTACK%' or
  V2Themes like '%WB_2457_CYBER_CRIME%'
```

#### Risk_028_massive_incident_of_data_fraud_and_theft
```
SELECT
  GKGRECORDID, DATE, Themes, V2Themes, V2Organizations , V2Persons, AllNames
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  V2Themes like '%TAX_FNCACT_HACKERS%' or
  V2Themes like '%TAX_FNCACT_HACKER%' or
  V2Themes like '%TAX_FNCACT_POLITICAL_HACKS%' or
  V2Themes like '%TAX_FNCACT_POLITICAL_HACK%' or
  V2Themes like '%TAX_TERROR_GROUP_REDHACK%' or
  V2Themes like '%TAX_FNCACT_BUSHWHACKER%' or
  V2Themes like '%TAX_FNCACT_BUSHWHACKERS%'
```
