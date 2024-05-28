package com.pets.pets_service.Models;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;

import java.util.List;

import javax.persistence.*;


@Data
@Entity
@DynamicUpdate
@DynamicInsert
@AllArgsConstructor
@NoArgsConstructor
public class ProductProvider {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String Name;
    private String email;
    private String address;
    private String phone;
    private String password;
    private Boolean locked = false;
    private Boolean enabled = false;

    @OneToMany(mappedBy = "productProviders")
    private List<Product> products;

    public ProductProvider(String Name,
                   String email,
                   String address,
                   String phone,
                   String password) {
        this.Name = Name;
        this.email = email;
        this.address = address;
        this.phone = phone;
        this.password = password;
    }
}
