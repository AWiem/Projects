package com.pets.pets_service.Service;

import com.pets.pets_service.Registration.token.PPConfirmationToken;
import com.pets.pets_service.Registration.token.PPConfirmationTokenService;
import com.pets.pets_service.Repositories.ProductProviderRepo;
import com.pets.pets_service.Models.ProductProvider;
import com.pets.pets_service.Models.PPCustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
@AllArgsConstructor
public class PPService implements UserDetailsService {

    private final static String USER_NOT_FOUND_MSG =
            "user with email %s not found";

    private final ProductProviderRepo pprepo;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;
    private final PPConfirmationTokenService confirmationTokenService;

    @Override
public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
    ProductProvider productprovider = pprepo.findByEmail(email)
            .orElseThrow(() -> new UsernameNotFoundException(String.format(USER_NOT_FOUND_MSG, email)));

    return new PPCustomUserDetails(productprovider);
}


    public String signUpUser(ProductProvider productprovider) {
        boolean userExists = pprepo
                .findByEmail(productprovider.getEmail())
                .isPresent();

        if (userExists) {
            // TODO check of attributes are the same and
            // TODO if email not confirmed send confirmation email.

            throw new IllegalStateException("email already taken");
        }

        String encodedPassword = bCryptPasswordEncoder
                .encode(productprovider.getPassword());

        productprovider.setPassword(encodedPassword);

        pprepo.save(productprovider);

        String token = UUID.randomUUID().toString();

        PPConfirmationToken confirmationToken = new PPConfirmationToken(
                token,
                LocalDateTime.now(),
                LocalDateTime.now().plusMinutes(15),
                productprovider
        );

        confirmationTokenService.saveConfirmationToken(
                confirmationToken);

//        TODO: SEND EMAIL
        
        return token;
    }

    public int enablePP(String email) {
        return pprepo.enablePP(email);
    }
}