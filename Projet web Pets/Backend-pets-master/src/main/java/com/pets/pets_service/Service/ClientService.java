package com.pets.pets_service.Service;

import com.pets.pets_service.Registration.token.ClientConfirmationToken;
import com.pets.pets_service.Registration.token.ClientConfirmationTokenService;
import com.pets.pets_service.Repositories.ClientRepo;
import com.pets.pets_service.Models.Client;
import com.pets.pets_service.Models.ClientCustomUserDetails;
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
public class ClientService implements UserDetailsService {

    private final static String USER_NOT_FOUND_MSG =
            "user with email %s not found";

    private final ClientRepo clientrepo;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;
    private final ClientConfirmationTokenService confirmationTokenService;

    @Override
public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
    Client client = clientrepo.findByEmail(email)
            .orElseThrow(() -> new UsernameNotFoundException(String.format(USER_NOT_FOUND_MSG, email)));

    return new ClientCustomUserDetails(client);
}


    public String signUpUser(Client client) {
        boolean userExists = clientrepo
                .findByEmail(client.getEmail())
                .isPresent();

        if (userExists) {
            // TODO check of attributes are the same and
            // TODO if email not confirmed send confirmation email.

            throw new IllegalStateException("email already taken");
        }

        String encodedPassword = bCryptPasswordEncoder
                .encode(client.getPassword());

        client.setPassword(encodedPassword);

        clientrepo.save(client);

        String token = UUID.randomUUID().toString();

        ClientConfirmationToken confirmationToken = new ClientConfirmationToken(
                token,
                LocalDateTime.now(),
                LocalDateTime.now().plusMinutes(15),
                client
        );

        confirmationTokenService.saveConfirmationToken(
                confirmationToken);

//        TODO: SEND EMAIL
        
        return token;
    }

    public int enableClient(String email) {
        return clientrepo.enableClient(email);
    }
}