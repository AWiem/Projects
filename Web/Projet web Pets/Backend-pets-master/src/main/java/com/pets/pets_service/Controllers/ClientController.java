package com.pets.pets_service.Controllers;
import com.pets.pets_service.Exception.ResourceNotFoundException;
import com.pets.pets_service.Models.Client;
import com.pets.pets_service.Repositories.ClientRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import javax.validation.Valid;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("api/v1")
public class ClientController {
    @Autowired
    private ClientRepo clientRepo;

    @GetMapping("/clients")
    public List<Client> getAllClients() {
        return clientRepo.findAll();
    }
    @GetMapping("/clients/{id}")
    public ResponseEntity<Client> getClientById(@PathVariable(value = "id") Integer clientId)
            throws ResourceNotFoundException {

        Client client = clientRepo.findById(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client not found"));
        return ResponseEntity.ok().body(client);
    }

    @GetMapping("/clients/email/{email}")
        public ResponseEntity<Client> getClientByEmail(@PathVariable(value = "email") String email)
            throws ResourceNotFoundException {

        Client client = clientRepo.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Client not found with email: " + email));
        return ResponseEntity.ok().body(client);
    }

 

    @PostMapping("/clients")
    public Client createClient(@Valid @RequestBody Client client) {
        return clientRepo.save(client);
    }

    @PutMapping("/clients/{id}")
    public ResponseEntity<Client> updateClient(@PathVariable(value = "id") Integer clientId,
                                               @Valid @RequestBody Client clientDetails) throws ResourceNotFoundException {
        Client client = clientRepo.findById(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client not found"));
        client.setFullName(clientDetails.getFullName());
        client.setId(clientDetails.getId());
        client.setPhone(clientDetails.getPhone());
        client.setAddress(clientDetails.getAddress());
        client.setEmail(clientDetails.getEmail());
        client.setPets(clientDetails.getPets());
        client.setAppointments(clientDetails.getAppointments());

        final Client updatedClient = clientRepo.save(client);
        return ResponseEntity.ok(updatedClient);
    }

    @DeleteMapping("/clients/{id}")
    public Map<String, Boolean> deleteClientAccount(@PathVariable(value = "id") Integer clientId)
            throws ResourceNotFoundException {
        Client client = clientRepo.findById(clientId)
                .orElseThrow(() -> new ResourceNotFoundException("Client not found"));
        clientRepo.delete(client);
        Map<String, Boolean> response = new HashMap<>();
        response.put("deleted", Boolean.TRUE);
        return response;
    }

}
