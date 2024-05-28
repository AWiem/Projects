package com.pets.pets_service.Service;

import com.pets.pets_service.Registration.token.VetConfirmationToken;
import com.pets.pets_service.Registration.token.VetConfirmationTokenService;
import com.pets.pets_service.Repositories.VeterinaryRepo;
import com.pets.pets_service.Models.Veterinary;
import com.pets.pets_service.Models.VetCustomUserDetails;
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
public class VetService implements UserDetailsService {

    private final static String USER_NOT_FOUND_MSG =
            "user with email %s not found";

    private final VeterinaryRepo vetrepo;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;
    private final VetConfirmationTokenService confirmationTokenService;

    @Override
public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
    Veterinary vet = vetrepo.findByEmail(email)
            .orElseThrow(() -> new UsernameNotFoundException(String.format(USER_NOT_FOUND_MSG, email)));

    return new VetCustomUserDetails(vet);
}


    public String signUpUser(Veterinary vet) {
        boolean userExists = vetrepo
                .findByEmail(vet.getEmail())
                .isPresent();

        if (userExists) {
            // TODO check of attributes are the same and
            // TODO if email not confirmed send confirmation email.

            throw new IllegalStateException("email already taken");
        }

        String encodedPassword = bCryptPasswordEncoder
                .encode(vet.getPassword());

        vet.setPassword(encodedPassword);

        vetrepo.save(vet);

        String token = UUID.randomUUID().toString();

        VetConfirmationToken confirmationToken = new VetConfirmationToken(
                token,
                LocalDateTime.now(),
                LocalDateTime.now().plusMinutes(15),
                vet
        );

        confirmationTokenService.saveConfirmationToken(
                confirmationToken);

//        TODO: SEND EMAIL
        
        return token;
    }

    public int enableVet(String email) {
        return vetrepo.enableVet(email);
    }
}