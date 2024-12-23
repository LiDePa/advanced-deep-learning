// clear old data
MATCH (n) DETACH DELETE n;

// nodes
CREATE (Person_70593c8d_998b_4548_9a97_440408a151f1:Person {role:'team blue', touching_ball:'False', touching_ground:'False', uid:'70593c8d_998b_4548_9a97_440408a151f1'})
CREATE (Person_5bc74850_86fe_4573_8b55_d08048f60a01:Person {role:'team blue', touching_ball:'False', touching_ground:'True', uid:'5bc74850_86fe_4573_8b55_d08048f60a01'})
CREATE (Person_d8948485_b2bd_433c_a74f_c386cdcd1bb7:Person {role:'team blue', touching_ball:'False', touching_ground:'True', uid:'d8948485_b2bd_433c_a74f_c386cdcd1bb7'})
CREATE (Person_46640c5b_9b1b_4e8e_8006_93402cb4994a:Person {role:'team black', touching_ball:'True', touching_ground:'False', uid:'46640c5b_9b1b_4e8e_8006_93402cb4994a'})
CREATE (Person_222755fe_dfe7_4041_993b_8979010932b0:Person {role:'ball boy', uid:'222755fe_dfe7_4041_993b_8979010932b0'})
CREATE (Person_687a7aac_6b2f_4643_a5e6_32c8f70a38bd:Person {role:'ball boy', uid:'687a7aac_6b2f_4643_a5e6_32c8f70a38bd'})
CREATE (Person_cd8aca9f_eeb4_497d_a6a0_018b51572838:Person {role:'viewer', uid:'cd8aca9f_eeb4_497d_a6a0_018b51572838'})
CREATE (Person_d3f75393_9a1e_41a0_8c19_bd1ceb892784:Person {role:'viewer', uid:'d3f75393_9a1e_41a0_8c19_bd1ceb892784'})
CREATE (Person_23eb5ecd_daee_4b1d_9fda_85752b8f5dfe:Person {role:'viewer', uid:'23eb5ecd_daee_4b1d_9fda_85752b8f5dfe'})
CREATE (Person_2420e13f_8b55_4301_bd52_70122e0509b4:Person {role:'viewer', uid:'2420e13f_8b55_4301_bd52_70122e0509b4'})
CREATE (Person_363df730_2479_4e5a_8e65_3f6d26e77aed:Person {role:'viewer', uid:'363df730_2479_4e5a_8e65_3f6d26e77aed'})
CREATE (Person_0a51e5fc_146b_424c_8f85_47fbb0675567:Person {role:'viewer', uid:'0a51e5fc_146b_424c_8f85_47fbb0675567'})
CREATE (Person_82349dd1_6eb4_4a87_88b8_3d393db44f43:Person {role:'viewer', uid:'82349dd1_6eb4_4a87_88b8_3d393db44f43'})
CREATE (Person_995a644b_6796_4028_8b21_634512899a29:Person {role:'viewer', uid:'995a644b_6796_4028_8b21_634512899a29'})
CREATE (Person_9d2c0355_78e3_4078_a7d8_ae96d6483682:Person {role:'viewer', uid:'9d2c0355_78e3_4078_a7d8_ae96d6483682'})
CREATE (Peron_c46f22c5_2b6c_4a87_941f_429243c9056b:Peron {role:'viewer', uid:'c46f22c5_2b6c_4a87_941f_429243c9056b'})
CREATE (Person_e9f8a6e0_421a_4d37_a91a_2bd364f88083:Person {role:'viewer', uid:'e9f8a6e0_421a_4d37_a91a_2bd364f88083'})
CREATE (Person_078394fc_c6bd_458d_8961_8f566be0d434:Person {role:'viewer', uid:'078394fc_c6bd_458d_8961_8f566be0d434'})
CREATE (Person_6862a49c_f7b9_41b8_808e_7284505eee7f:Person {role:'viewer', uid:'6862a49c_f7b9_41b8_808e_7284505eee7f'})
CREATE (Person_61adffef_7b67_40c0_a37e_4981521eef17:Person {role:'viewer', uid:'61adffef_7b67_40c0_a37e_4981521eef17'})
CREATE (Ball_f191014a_9225_4800_90eb_90563f248972:Ball {touching_ground:'False', uid:'f191014a_9225_4800_90eb_90563f248972'})
CREATE (Ground_e3b712ec_efff_415e_952c_221fdab52b06:Ground {playing_field:'False', uid:'e3b712ec_efff_415e_952c_221fdab52b06'})
CREATE (Ground_1dc4b828_681b_42d7_bc6e_d7eb77b23816:Ground {playing_field:'True', uid:'1dc4b828_681b_42d7_bc6e_d7eb77b23816'})
CREATE (Stairs_34860ada_e7a3_474b_bca2_220e4bea7996:Stairs {uid:'34860ada_e7a3_474b_bca2_220e4bea7996'})
CREATE (Advertisement_640adec8_9338_48b9_91fc_4c0ae5e97a97:Advertisement {advertiser:'KAI OS EVIZIO', uid:'640adec8_9338_48b9_91fc_4c0ae5e97a97'})
CREATE (Advertisement_80b96f49_948e_4d96_bb02_0688cfdeddd5:Advertisement {advertiser:'DRV', uid:'80b96f49_948e_4d96_bb02_0688cfdeddd5'})
CREATE (Advertisement_f49062ed_221b_46f4_b3ed_684e2b188f2f:Advertisement {advertiser:'AIKA', uid:'f49062ed_221b_46f4_b3ed_684e2b188f2f'})

// relations

// Only the player of team black is touching the ball
MATCH (p:Person {role: "team black", uid: "46640c5b_9b1b_4e8e_8006_93402cb4994a"}), (b:Ball)
CREATE (p)-[:TOUCHING]->(b);

// All players with attribute "touching_ground": True are touching the ground
MATCH (p:Person {touching_ground: "True"}), (g:Ground)
CREATE (p)-[:TOUCHING]->(g);

// Only the person with uid: "363df730_2479_4e5a_8e65_3f6d26e77aed" is touching the stairs
MATCH (p:Person {uid: "363df730_2479_4e5a_8e65_3f6d26e77aed"}), (s:Stairs)
CREATE (p)-[:TOUCHING]->(s);

// The persons with role "ball boy" are touching the ground that has attribute "playing_field": False
MATCH (p:Person {role: "ball boy"}), (g:Ground {playing_field: "False"})
CREATE (p)-[:TOUCHING]->(g);
