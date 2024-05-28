package com.pets.pets_service.Registration;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Data;
import lombok.ToString;

@Data
@AllArgsConstructor
@EqualsAndHashCode
@ToString
public class PPRegistrationRequest {
    private final String fullName;
    private final String email;
    private final String address;
    private final String phone;
    private final String password;
}
