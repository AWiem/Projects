package com.pets.pets_service.Models;
import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Data;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;

import javax.persistence.*;

@Data
@Entity
@DynamicUpdate
@DynamicInsert
@Table(name = "pet")
public class Pet {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String sex;
    private String age;
    private String furColor;
    private String imgUrl;
    private boolean forAdoption;
    @Enumerated(EnumType.STRING)
    private PetType petType;

    @ManyToOne
    @JsonIgnore
    private Client client;



}

