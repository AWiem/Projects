package com.pets.pets_service.Models;


import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;

public class PPCustomUserDetails implements UserDetails {

    private final ProductProvider productprovider;

    public PPCustomUserDetails(ProductProvider productprovider) {
        this.productprovider = productprovider;
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        // return authorities if needed
        return null;
    }

    @Override
    public String getPassword() {
        return productprovider.getPassword();
    }

    @Override
    public String getUsername() {
        return productprovider.getEmail();
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true; // change this based on the productprovider's enabled status
    }
}
