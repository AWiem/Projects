package com.pets.pets_service.Models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;

import java.util.Set;

import javax.persistence.*;

@Data
@Entity
@DynamicUpdate
@DynamicInsert
@AllArgsConstructor
@NoArgsConstructor
public class Veterinary{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String fullName;
    private String email;
    private String phone;
    private String officeAddress;
    private String speciality;
    private String password;
    private Boolean locked = false;
    private Boolean enabled = false;

    @OneToMany(cascade = CascadeType.ALL,mappedBy = "veterinary")
    private Set<Appointment> appointments;

    public Veterinary(String fullName,
                   String email,
                   String officeAddress,
                   String phone,
                   String speciality,
                   String password) {
        this.fullName = fullName;
        this.email = email;
        this.officeAddress = officeAddress;
        this.phone = phone;
        this.speciality = speciality;
        this.password = password;
    }
}
